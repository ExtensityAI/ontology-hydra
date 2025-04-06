import logging
import secrets
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ontopipe.cqs.comittee import Comittee, ComitteeMember, generate_comittee_for_domain
from ontopipe.cqs.question_generation import generate_questions
from ontopipe.cqs.scoping import generate_scope_document, merge_scope_documents

DOMAINS = [
    ("antarctica", "Antarctica"),
]


def _init_logger(path: Path):
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    file = logging.FileHandler(path / f"eval_{now}.log")
    file.setFormatter(formatter)
    logger.addHandler(file)

    return logger


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

    logger = _init_logger(eval_path)
    config = EvalConfig(eval_path=eval_path, run_id=args.run_id, logger=logger)

    config.logger.info(
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
            logger=logger.getChild(id),
            paths=paths,
            config=config,
        )

        eval(params)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Config across the whole valuation run"""

    eval_path: Path
    logger: logging.Logger
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

    logger: logging.Logger

    paths: EvalPaths
    config: EvalConfig


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
    params.logger.info("Generated CQs for %s", params.domain)


def _generate_comittee(params: EvalParams):
    params.logger.debug("Generating comittee for %s", params.domain)

    if params.paths.comittee_path.exists():
        comittee = Comittee.model_validate_json(
            params.paths.comittee_path.read_text(encoding="utf-8")
        )
    else:
        comittee = generate_comittee_for_domain(params.domain)

        params.paths.comittee_path.write_text(
            comittee.model_dump_json(indent=2), encoding="utf-8"
        )

    return comittee


def _generate_scope_documents(groups: list[list[ComitteeMember]], params: EvalParams):
    params.logger.debug("Generating scope documents for %s", params.domain)

    scope_documents = []

    # generate or load scope documents for each group
    for i, group in enumerate(groups):
        group_scope_document_path = params.paths.scopes_path / f"scope_group_{i}.txt"

        if group_scope_document_path.exists():
            scope = group_scope_document_path.read_text(encoding="utf-8")
        else:
            scope = generate_scope_document(params.domain, [m.persona for m in group])

            group_scope_document_path.write_text(scope, encoding="utf-8")

        scope_documents.append(scope)

    return scope_documents


def _merge_scope_documents(scope_documents: list[str], params: EvalParams):
    params.logger.debug("Merging scope documents for %s", params.domain)

    if params.paths.merged_scope_path.exists():
        merged_scope = params.paths.merged_scope_path.read_text(encoding="utf-8")
    else:
        merged_scope = merge_scope_documents(params.domain, scope_documents)
        params.paths.merged_scope_path.write_text(merged_scope, encoding="utf-8")

    return merged_scope


def _generate_all_cqs(groups: list[list[ComitteeMember]], params: EvalParams):
    params.logger.debug("Generating all CQs for %s", params.domain)

    all_cqs = []
    for i, group in enumerate(groups):
        group_cqs_path = params.paths.cqs_path / f"cqs_group_{i}.txt"

        if group_cqs_path.exists():
            group_cqs = group_cqs_path.read_text(encoding="utf-8").split("\n")
        else:
            group_cqs = generate_questions(
                params.domain,
                group,
                params.paths.merged_scope_path.read_text(encoding="utf-8"),
            )

            group_cqs_path.write_text("\n".join(group_cqs), encoding="utf-8")

        all_cqs.extend(group_cqs)

    params.paths.all_cqs_path.write_text("\n".join(all_cqs), encoding="utf-8")

    return all_cqs


def _generate_cqs(params: EvalParams):
    params.logger.info("Generating CQs for %s", params.domain)

    comittee = _generate_comittee(params)
    params.logger.debug("Comittee has %s members", len(comittee.members))

    groups = comittee.divide_into_groups(4)

    scope_documents = _generate_scope_documents(groups, params)
    params.logger.debug("Generated %i scope documents", len(scope_documents))

    merged_scope = _merge_scope_documents(scope_documents, params)
    params.logger.debug("Merged scope document has %i characters", len(merged_scope))

    cqs = _generate_all_cqs(groups, params)
    params.logger.debug("Generated %i CQs", len(cqs))

    return cqs
