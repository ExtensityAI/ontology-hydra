import json
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
from ontopipe.kg.kg_generation import generate_kg
from ontopipe.models import KG, Ontology
from ontopipe.ontology.ontology_generation import generate_ontology

DOMAINS = [
    ("To_Kill_A_Mockingbird", "To Kill A Mockingbird"),
]

_colored_log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {extra[domain_id]} | <level>{message}</level>"

_uncolored_log_format = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS}| {level: <8} | {extra[domain_id]} | {message}"
)


def _init_logger(path: Path):
    log_path = path / "logs" / "eval_{time}.log"
    log_path.parent.mkdir(exist_ok=True, parents=True)

    def is_ontopipe_log(record):
        return "ontopipe" in record["name"]

    logger.remove()

    logger.configure(extra={"domain_id": "-"})

    logger.add(log_path, rotation="300 MB", level="DEBUG", format=_uncolored_log_format)
    logger.add(
        sys.stderr, level="DEBUG", filter=is_ontopipe_log, format=_colored_log_format
    )


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


def _find_domain_train_data(train_data: dict, domain_id: str):
    for entry in train_data["data"]:
        if entry["title"].lower() == domain_id.lower():
            return entry

    raise ValueError(f"No training data found for domain {domain_id}")


def main():
    args = _parse_args()
    eval_path = Path(args.path) / "runs" / args.run_id
    eval_path.mkdir(exist_ok=True, parents=True)

    train_file_path = args.path / "train-v2.0.json"

    if not train_file_path.exists():
        logger.error(
            "Training file not found at {}! Please read the README.md in the eval directory!>",
            train_file_path,
        )
        sys.exit(1)

    _init_logger(eval_path)
    config = EvalConfig(
        eval_path=eval_path,
        train_data=json.loads(train_file_path.read_text(encoding="utf-8")),
        run_id=args.run_id,
    )

    logger.info(
        "Starting evaluation {} under {}",
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
            domain_train_data=_find_domain_train_data(config.train_data, id),
            logger=logger.bind(domain_id=id),
            paths=paths,
            config=config,
        )

        eval(params)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Config across the whole valuation run"""

    eval_path: Path
    train_data: dict
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

    domain_train_data: dict

    logger: "loguru.Logger"

    paths: EvalPaths
    config: EvalConfig


def _read_comittee_if_cached(path: Path):
    """Read the comittee from the cache if it exists, otherwise return None"""

    if path.exists():
        return Comittee.model_validate_json(path.read_text(encoding="utf-8"))


def _generate_comittee(params: EvalParams):
    params.logger.debug("Generating comittee for {}", params.domain)

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
    params.logger.debug("Generating CQs...", params.domain)

    comittee = _generate_comittee(params)
    params.logger.debug("Generated comittee with {} members", len(comittee.members))

    groups = comittee.divide_into_groups(4)

    scope_documents = _generate_scope_documents(groups, params)
    params.logger.debug("Generated {} scope documents", len(scope_documents))

    merged_scope = _merge_scope_documents(scope_documents, params)
    params.logger.debug("Merged scope document has {} characters", len(merged_scope))

    return _generate_all_cqs(groups, params)


def _generate_ontology(cqs: list[str], path: Path, params: EvalParams):
    params.logger.debug("Generating ontology...")

    ontology_folder_path = path / "ontology"
    ontology_folder_path.mkdir(exist_ok=True, parents=True)

    ontology_file_path = ontology_folder_path / "ontology.json"

    if ontology_file_path.exists():
        return Ontology.model_validate_json(
            ontology_file_path.read_text(encoding="utf-8")
        ), ontology_file_path

    else:
        return generate_ontology(
            cqs,
            params.domain,
            folder=ontology_folder_path,
            fname=ontology_file_path.name,
        ), ontology_file_path


def _generate_kg(
    texts: list[str], path: Path, ontology_file_path: Path, params: EvalParams
):
    params.logger.debug("Generating KG...")

    kg_folder_path = path / "kg"
    kg_folder_path.mkdir(exist_ok=True, parents=True)

    kg_file_path = kg_folder_path / "kg.json"

    if kg_file_path.exists():
        return KG.model_validate_json(kg_file_path.read_text(encoding="utf-8"))

    else:
        return generate_kg(
            texts,
            params.domain,
            ontology_file=ontology_file_path,
            output_folder=kg_folder_path,
            output_filename=kg_file_path.name,
        )


def eval(params: EvalParams):
    """Run evaluation for a single domain"""

    params.logger.info(
        "Evaluating domain {domain} ({id})",
        domain=params.domain,
        id=params.id,
    )

    path = params.config.eval_path / params.id
    path.mkdir(exist_ok=True, parents=True)

    cqs = _generate_cqs(params)
    params.logger.info("Generated {} CQs", len(cqs))

    ontology, ontology_file_path = _generate_ontology(cqs, path, params)
    params.logger.info(
        "Generated ontology with {} subclass relations, {} object properties, and {} data properties",
        len(ontology.subclass_relations),
        len(ontology.object_properties),
        len(ontology.data_properties),
    )

    # take contexts from the training data
    texts = [p["context"] for p in params.domain_train_data["paragraphs"]]

    params.logger.debug(
        "Found {} contexts in training data with a total length of ~{} words",
        len(texts),
        sum(len(t.split(" ")) for t in texts),
    )

    kg = _generate_kg(
        texts=texts,
        path=path,
        ontology_file_path=ontology_file_path,
        params=params,
    )

    params.logger.info("Generated KG with {} triplets", len(kg.triplets))
