import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from pathlib import Path
from typing import Literal, cast

from symai.components import MetadataTracker

from ontopipe.cqs.comittee import Comittee, generate_comittee_for_domain
from ontopipe.cqs.question_generation import (
    Duplicates,
    Question,
    QuestionDeduplicator,
    Questions,
    generate_questions,
)
from ontopipe.cqs.scoping import generate_scope_document, merge_scope_documents
from ontopipe.ontology.generator import generate_ontology
from ontopipe.ontology.models import Ontology

logger = getLogger("ontopipe.pipe")
# use standard logging module as ontopipe is a tool/library and we do not want to enforce a specific logging library on users


def _generate_comittee_with_cache(domain: str, cache_path: Path):
    if cache_path.exists():
        return Comittee.model_validate_json(cache_path.read_text(encoding="utf-8", errors="ignore"))

    comittee = generate_comittee_for_domain(domain)
    cache_path.write_text(comittee.model_dump_json(indent=2), encoding="utf-8")
    return comittee


def _generate_scope_documents_with_cache(domain: str, comittee: Comittee, cache_path: Path, group_size):
    groups = comittee.divide_into_groups(group_size)
    documents = [None] * len(groups)

    def process_group(i_group):
        i, group = i_group
        doc_cache_path = cache_path / f"scope_{i}.txt"
        if doc_cache_path.exists():
            return i, doc_cache_path.read_text(encoding="utf-8", errors="ignore")
        doc = generate_scope_document(domain, [m.persona for m in group])
        doc_cache_path.write_text(doc, encoding="utf-8")
        return i, doc

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_group, enumerate(groups))
        for i, doc in results:
            documents[i] = doc

    return cast(list[str], documents)


def _merge_scope_documents_with_cache(domain: str, documents: list[str], cache_path: Path):
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    merged_scope = merge_scope_documents(domain, documents)
    cache_path.write_text(merged_scope, encoding="utf-8")
    return merged_scope


def _generate_scope_with_cache(domain: str, comittee: Comittee, group_size: int, cache_path: Path):
    merged_scope_path = cache_path / "scope_merged.txt"

    # in case the merged scope exists, we can load it directly and skip everything else
    if merged_scope_path.exists():
        return merged_scope_path.read_text(encoding="utf-8", errors="ignore")

    documents = _generate_scope_documents_with_cache(domain, comittee, cache_path, group_size)

    return _merge_scope_documents_with_cache(domain, documents, merged_scope_path)


def _sort_cqs(cqs: Iterable[str]):
    """Sorts CQs by length of the question, longest first."""
    return sorted(cqs, key=lambda x: len(x.split(" ")), reverse=True)


def _deduplicate_cqs(cqs: list[str], cache_path: Path) -> list[str]:
    cqs = _sort_cqs(set(cqs))
    questions = Questions(items=[Question(index=i, text=q) for i, q in enumerate(cqs)])

    with MetadataTracker() as tracker:
        deduplicator = QuestionDeduplicator()
        res: Duplicates = deduplicator(input=questions)

        deduplicator.contract_perf_stats()
        logger.debug("CQ Deduplication API Usage: %s", tracker.usage)

    # TODO write duplicates to cache file, but not only with indexes but actual questions!
    cache_path.write_text(res.model_dump_json(indent=2), encoding="utf-8")

    # deduplicate CQs based on the duplicates found (1. take new questions and 2. add all non-duplicates)
    deduplicated_cqs = set(d.question for d in res.duplicates)
    deduplicated_cqs.update(set(cqs) - set(cqs[i] for d in res.duplicates for i in d.indexes))

    len_before = len(cqs)
    cqs = _sort_cqs(deduplicated_cqs)
    logger.debug("Deduplicated %d CQs to %d unique CQs", len_before, len(cqs))
    return cqs


def _generate_cqs_with_cache(
    domain: str,
    merged_scope: str,
    group_size: int,
    comittee: Comittee,
    cache_path: Path,
):
    combined_cqs_path = cache_path / "cqs_combined.txt"

    # in case all cqs were generated and combined, we can load them directly and skip everything else
    if combined_cqs_path.exists():
        return combined_cqs_path.read_text(encoding="utf-8", errors="ignore").split("\n")

    groups = list(comittee.divide_into_groups(group_size))
    cqs = [None] * len(groups)

    def process_group(i_group):
        i, group = i_group
        group_cqs_cache_path = cache_path / f"cqs_{i}.txt"
        if group_cqs_cache_path.exists():
            return i, group_cqs_cache_path.read_text(encoding="utf-8", errors="ignore").split("\n")
        group_cqs = generate_questions(domain, group, merged_scope)
        group_cqs_cache_path.write_text("\n".join(group_cqs), encoding="utf-8")
        return i, group_cqs

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_group, enumerate(groups))
        for i, group_cqs in results:
            cqs[i] = group_cqs

    # flatten the list of lists
    cqs = [cq for gcq in cqs for cq in gcq]

    # deduplicate CQs
    cqs = _deduplicate_cqs(cqs, cache_path / "duplicates.json")

    combined_cqs_path.write_text("\n".join(cqs), encoding="utf-8")

    return cqs


def _generate_ontology_with_cache(
    cqs: list[str],
    cache_path: Path,
    cqs_per_batch: int = 4,
):
    if cache_path.exists():
        ontology = Ontology.model_validate_json(cache_path.read_text(encoding="utf-8", errors="ignore"))

    else:
        logger.debug("Generating ontology from %d CQs", len(cqs))
        ontology = generate_ontology(cqs, cache_path, cqs_per_batch=cqs_per_batch)

    return ontology


def ontopipe(
    domain: str,
    group_size: int = 4,
    cqs_per_batch: int = 4,
    cache_path: Path | Literal["use_temp"] = "use_temp",
):
    """Runs the ontopipe pipeline to generate an ontology for the given domain.

    Args:
        domain (str): The domain for which to generate the ontology.
        group_size (int): The number of committee members to group together for scope and cq generation. Defaults to 4.
        cqs_per_batch (int): The number of CQs to process in a single batch during ontology generation. Defaults to 4.
        cache_path (Path): The path to the cache directory. If it does not exist, a temp directory will be created."""

    if cache_path == "use_temp":
        # use temp path for caching
        cache_path = Path(tempfile.mkdtemp("ontopipe"))

    if not cache_path.exists() or not cache_path.is_dir():
        raise ValueError(f"Cache path '{cache_path}' is not a directory or does not exist")

    logger.debug("Generating ontology for domain: '%s'", domain)
    logger.debug("Using cache path: %s", cache_path)

    comittee_path = cache_path / "comittee.json"
    scopes_path: Path = cache_path / "scopes"
    cqs_path = cache_path / "cqs"
    ontology_path = cache_path / "ontology.json"

    scopes_path.mkdir(exist_ok=True, parents=True)
    cqs_path.mkdir(exist_ok=True, parents=True)

    comittee = _generate_comittee_with_cache(domain, comittee_path)
    logger.debug(
        "Generated comittee for domain '%s' with %d members",
        domain,
        len(comittee.members),
    )

    scope = _generate_scope_with_cache(domain, comittee, group_size, scopes_path)
    logger.debug("Generated scope for domain '%s' with %d words", domain, len(scope.split(" ")))

    cqs = _generate_cqs_with_cache(domain, scope, group_size, comittee, cqs_path)
    logger.debug("Generated %d CQs for domain '%s'", len(cqs), domain)

    ontology = _generate_ontology_with_cache(cqs, ontology_path, cqs_per_batch=cqs_per_batch)

    logger.debug(
        "Generated ontology for domain '%s' with %d classes and %d properties",
        domain,
        len(ontology.classes),
        len(ontology.properties),
    )
    return ontology
