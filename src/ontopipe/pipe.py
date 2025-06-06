import os
import tempfile
from logging import getLogger
from pathlib import Path

from symai.components import DynamicEngine, MetadataTracker

from ontopipe.cqs.comittee import Comittee, generate_comittee_for_domain
from ontopipe.cqs.question_generation import QuestionDeduplicator, Questions, generate_questions
from ontopipe.cqs.scoping import generate_scope_document, merge_scope_documents
from ontopipe.models import Ontology
from ontopipe.ontology.ontology_fixing import fix_ontology
from ontopipe.ontology.ontology_generation import generate_ontology
from ontopipe.vis import visualize_ontology

logger = getLogger("ontopipe.pipe")
# use standard logging module as ontopipe is a tool/library and we do not want to enforce a specific logging library on users


def _generate_comittee_with_cache(domain: str, cache_path: Path):
    if cache_path.exists():
        return Comittee.model_validate_json(cache_path.read_text())

    comittee = generate_comittee_for_domain(domain)
    cache_path.write_text(comittee.model_dump_json(indent=2))
    return comittee


def _generate_scope_documents_with_cache(domain: str, comittee: Comittee, cache_path: Path):
    groups = comittee.divide_into_groups(4)  # TODO n is hyperparam
    documents = []

    for i, group in enumerate(groups):
        doc_cache_path = cache_path / f"scope_{i}.txt"

        if doc_cache_path.exists():
            documents.append(doc_cache_path.read_text())
            continue

        doc = generate_scope_document(domain, [m.persona for m in group])
        documents.append(doc)

        doc_cache_path.write_text(doc)

    return documents


def _merge_scope_documents_with_cache(domain: str, documents: list[str], cache_path: Path):
    if cache_path.exists():
        return cache_path.read_text()

    merged_scope = merge_scope_documents(domain, documents)
    cache_path.write_text(merged_scope)
    return merged_scope


def _generate_scope_with_cache(domain: str, comittee: Comittee, cache_path: Path):
    merged_scope_path = cache_path / "scope_merged.txt"

    # in case the merged scope exists, we can load it directly and skip everything else
    if merged_scope_path.exists():
        return merged_scope_path.read_text()

    documents = _generate_scope_documents_with_cache(domain, comittee, cache_path)

    return _merge_scope_documents_with_cache(domain, documents, merged_scope_path)


def _deduplicate_cqs(cqs: list[str]) -> list[str]:
    cqs = list(set(cqs))
    cqs = sorted(cqs, key=lambda x: len(x.split(" ")), reverse=True)

    with MetadataTracker() as tracker:
        deduplicator = QuestionDeduplicator()
        deduplicated_cqs = deduplicator(input=Questions(items=cqs))

        deduplicator.contract_perf_stats()
        logger.debug("CQ Deduplication API Usage: %s", tracker.usage)

    logger.debug("Deduplicated %d CQs to %d unique CQs", len(cqs), len(deduplicated_cqs.items))

    return deduplicated_cqs.items


def _generate_cqs_with_cache(domain: str, merged_scope: str, comittee: Comittee, cache_path: Path):
    combined_cqs_path = cache_path / "cqs_combined.txt"

    # in case all cqs were generated and combined, we can load them directly and skip everything else
    if combined_cqs_path.exists():
        return combined_cqs_path.read_text().split("\n")

    cqs = []

    for i, group in enumerate(comittee.divide_into_groups(4)):  # TODO n is hyperparam
        group_cqs_cache_path = cache_path / f"cqs_{i}.txt"

        if group_cqs_cache_path.exists():
            cqs.extend(group_cqs_cache_path.read_text().split("\n"))
            continue

        group_cqs = generate_questions(domain, group, merged_scope)
        group_cqs_cache_path.write_text("\n".join(group_cqs))
        cqs.extend(group_cqs)

    # deduplicate CQs
    cqs = _deduplicate_cqs(cqs)

    combined_cqs_path.write_text("\n".join(cqs))

    return cqs


def _generate_ontology_with_cache(domain: str, cqs: list[str], cache_path: Path, fixed_cache_path: Path):
    if fixed_cache_path.exists():
        # we have a cached fixed ontology, load it directly
        return Ontology.model_validate_json(fixed_cache_path.read_text())

    if cache_path.exists():
        ontology = Ontology.model_validate_json(cache_path.read_text())

    else:
        logger.debug("Generating ontology from %d CQs", len(cqs))
        ontology = generate_ontology(cqs, domain, cache_path.parent, cache_path.name, batch_size=4)

    logger.debug("Fixing ontology")
    ontology = fix_ontology(ontology, fixed_cache_path.parent, fixed_cache_path.name)
    fixed_cache_path.write_text(ontology.model_dump_json(indent=2))

    visualize_ontology(ontology, Path(str(cache_path) + ".html"))

    return ontology


def ontopipe(domain: str, cache_path: Path = Path(tempfile.mkdtemp("ontopipe"))):
    """Runs the ontopipe pipeline to generate an ontology for the given domain.

    Args:
        domain (str): The domain for which to generate the ontology.
        cache_path (Path): The path to the cache directory. If it does not exist, a temp directory will be created."""

    if not cache_path.exists() or not cache_path.is_dir():
        raise ValueError(f"Cache path '{cache_path}' is not a directory or does not exist")

    logger.debug("Generating ontology for domain: '%s'", domain)
    logger.debug("Using cache path: %s", cache_path)

    comittee_path = cache_path / "comittee.json"
    scopes_path: Path = cache_path / "scopes"
    cqs_path = cache_path / "cqs"
    ontology_path = cache_path / "ontology.json"
    fixed_ontology_path = cache_path / "ontology_fixed.json"

    scopes_path.mkdir(exist_ok=True, parents=True)
    cqs_path.mkdir(exist_ok=True, parents=True)

    comittee = _generate_comittee_with_cache(domain, comittee_path)
    logger.debug(
        "Generated comittee for domain '%s' with %d members",
        domain,
        len(comittee.members),
    )

    scope = _generate_scope_with_cache(domain, comittee, scopes_path)
    logger.debug("Generated scope for domain '%s' with %d words", domain, len(scope.split(" ")))

    cqs = _generate_cqs_with_cache(domain, scope, comittee, cqs_path)
    logger.debug("Generated %d CQs for domain '%s'", len(cqs), domain)

    # use o3-mini for ontology generation
    # TODO make this configurable
    with DynamicEngine("o4-mini", os.getenv("NEUROSYMBOLIC_ENGINE_API_KEY")):
        ontology = _generate_ontology_with_cache(domain, cqs, ontology_path, fixed_ontology_path)

    logger.debug(
        "Generated ontology for domain '%s' with %d subclass relations",
        domain,
        len(ontology.subclass_relations),
    )
    return ontology
