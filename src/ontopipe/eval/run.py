import secrets
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import loguru
from loguru import logger

from ontopipe.cqs.comittee import Comittee, ComitteeMember, generate_comittee_for_domain
from ontopipe.cqs.question_generation import generate_questions
from ontopipe.cqs.scoping import generate_scope_document, merge_scope_documents
from ontopipe.models import Ontology
from ontopipe.ontology.ontology_generation import generate_ontology

DOMAINS = [
    ("antarctica", "Antarctica"),
]


def _init_logger(path: Path):
    log_path = path / "logs" / "eval_{time}.log"
    log_path.parent.mkdir(exist_ok=True, parents=True)

    logger.add(log_path, rotation="100 MB", level="DEBUG")
    logger.add(sys.stderr, level="DEBUG")


def _parse_args():
    parser = ArgumentParser(description="Run evaluation")

    parser.add_argument(
        "--path",
        type=Path,
        help="Base path for evaluation",
        default=Path("eval"),
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID for evaluation",
        default=secrets.token_urlsafe(6),
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    eval_path = args.path / "runs" / args.run_id
    eval_path.mkdir(exist_ok=True, parents=True)

    _init_logger(eval_path)
    config = EvalConfig(eval_path=eval_path, run_id=args.run_id)

    logger.info(
        "Starting evaluation %s under %s",
        config.run_id,
        config.eval_path,
    )

    # run eval for each domain
    for id, domain in DOMAINS:
        path = config.eval_path / id
        path.mkdir(exist_ok=True, parents=True)

        paths = EvalPaths(
            comittee_path=path / "comittee.json",
            scopes_path=path / "scopes",
            merged_scope_path=path / "merged_scope.txt",
            cqs_path=path / "cqs",
            all_cqs_path=path / "all_cqs.txt",
        )

        paths.scopes_path.mkdir(exist_ok=True, parents=True)
        paths.cqs_path.mkdir(exist_ok=True, parents=True)

        params = EvalParams(
            id=id,
            domain=domain,
            logger=logger.bind(domain=id),
            paths=paths,
            config=config,
        )

        eval(params)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Config across the whole valuation run"""

    eval_path: Path
    run_id: str


@dataclass(frozen=True, slots=True)
class EvalPaths:
    """Paths for a single evaluation"""

    comittee_path: Path
    scopes_path: Path
    cqs_path: Path
    merged_scope_path: Path
    all_cqs_path: Path


@dataclass(frozen=True, slots=True)
class EvalParams:
    """Parameters for a single evaluation"""

    id: str
    domain: str

    logger: "loguru.Logger"

    paths: EvalPaths
    config: EvalConfig


def _read_comittee_if_cached(path: Path):
    """Read the comittee from the cache if it exists, otherwise return None"""

    if path.exists():
        return Comittee.model_validate_json(path.read_text(encoding="utf-8"))


def _generate_comittee(params: EvalParams):
    params.logger.debug("Generating comittee for %s", params.domain)

    comittee = _read_comittee_if_cached(params.paths.comittee_path)

    if not comittee:
        comittee = generate_comittee_for_domain(params.domain)

        params.paths.comittee_path.write_text(
            comittee.model_dump_json(indent=2), encoding="utf-8"
        )

    return comittee


def _generate_scope_documents(groups: list[list[ComitteeMember]], params: EvalParams):
    params.logger.debug("Generating scope documents...")

    scope_documents = []

    # generate or load scope documents for each group
    for i, group in enumerate(groups):
        path = params.paths.scopes_path / f"scope_group_{i}.txt"

        # read cached scope document or else generate it
        if path.exists():
            scope = path.read_text(encoding="utf-8")
        else:
            scope = generate_scope_document(params.domain, [m.persona for m in group])

            path.write_text(scope, encoding="utf-8")

        scope_documents.append(scope)

    return scope_documents


def _merge_scope_documents(scope_documents: list[str], params: EvalParams):
    params.logger.debug("Merging scope documents...")

    path = params.paths.merged_scope_path

    # read cached merged scope document or else generate it
    if path.exists():
        merged_scope = path.read_text(encoding="utf-8")
    else:
        merged_scope = merge_scope_documents(params.domain, scope_documents)
        path.write_text(merged_scope, encoding="utf-8")

    return merged_scope


def _generate_all_cqs(groups: list[list[ComitteeMember]], params: EvalParams):
    path = params.paths.all_cqs_path

    # read cached list of all CQs or else generate them
    if path.exists():
        return path.read_text(encoding="utf-8").split("\n")

    all_cqs = []
    for i, group in enumerate(groups):
        path = params.paths.cqs_path / f"cqs_group_{i}.txt"

        if path.exists():
            group_cqs = path.read_text(encoding="utf-8").split("\n")
        else:
            group_cqs = generate_questions(
                params.domain,
                group,
                params.paths.merged_scope_path.read_text(encoding="utf-8"),
            )

            path.write_text("\n".join(group_cqs), encoding="utf-8")

        all_cqs.extend(group_cqs)

    path.write_text("\n".join(all_cqs), encoding="utf-8")

    return all_cqs


def _generate_cqs(params: EvalParams):
    params.logger.info("Generating CQs...", params.domain)

    comittee = _generate_comittee(params)
    params.logger.debug("Generated comittee with %s members", len(comittee.members))

    groups = comittee.divide_into_groups(4)

    scope_documents = _generate_scope_documents(groups, params)
    params.logger.debug("Generated %i scope documents", len(scope_documents))

    merged_scope = _merge_scope_documents(scope_documents, params)
    params.logger.debug("Merged scope document has %i characters", len(merged_scope))

    return _generate_all_cqs(groups, params)


def _generate_ontology(cqs: list[str], path: Path, params: EvalParams):
    params.logger.info("Generating ontology...")

    ontology_folder_path = path / "ontology"
    ontology_folder_path.mkdir(exist_ok=True, parents=True)

    ontology_file_path = ontology_folder_path / "ontology.json"

    if ontology_file_path.exists():
        return Ontology.model_validate_json(
            ontology_file_path.read_text(encoding="utf-8")
        )

    else:
        return generate_ontology(
            cqs,
            params.domain,
            folder=ontology_folder_path,
            fname=ontology_file_path.name,
        )


def eval(params: EvalParams):
    """Run evaluation for a single domain"""

    params.logger.info(
        "Starting evaluation %s for domain %s",
        params.id,
        params.domain,
    )

    path = params.config.eval_path / params.id
    path.mkdir(exist_ok=True, parents=True)

    cqs = _generate_cqs(params)
    params.logger.debug("Generated %i CQs", len(cqs))
