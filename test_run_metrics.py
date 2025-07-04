#!/usr/bin/env python3
"""
Test script to verify the new run metrics and runtime statistics functionality.
"""

import json
import tempfile
from pathlib import Path
from packages.eval.src.eval.eval import _generate_run_metrics_and_stats
from packages.eval.src.eval.eval import EvalScenario
from packages.eval.src.eval.neo4j_eval import Neo4jConfig

def create_mock_evaluation_data(base_path: Path):
    """Create mock evaluation data to test the metrics aggregation."""

    # Create a mock scenario
    scenario_path = base_path / "biomedical_engineer"
    scenario_path.mkdir(parents=True, exist_ok=True)

    # Create mock ontology runtime stats
    ontology_stats = {
        'total_elapsed_time_seconds': 120.5,
        'total_prompt_tokens': 5000,
        'total_completion_tokens': 2000,
        'total_tokens': 7000,
        'total_calls': 15
    }
    with open(scenario_path / "ontology_runtime_stats.json", 'w') as f:
        json.dump(ontology_stats, f)

    # Create mock topics
    topics_path = scenario_path / "topics"
    topics_path.mkdir(exist_ok=True)

    topic_path = topics_path / "Biomedical Engineer"
    topic_path.mkdir(exist_ok=True)

    # Create mock KG runtime stats
    kg_stats = {
        'total_elapsed_time_seconds': 45.2,
        'total_prompt_tokens': 3000,
        'total_completion_tokens': 1500,
        'total_tokens': 4500,
        'total_calls': 8
    }
    with open(topic_path / "kg_runtime_stats.json", 'w') as f:
        json.dump(kg_stats, f)

    # Create mock QA runtime stats
    qa_stats = {
        'total_elapsed_time_seconds': 30.1,
        'total_prompt_tokens': 2000,
        'total_completion_tokens': 1000,
        'total_tokens': 3000,
        'total_calls': 5,
        'total_questions_processed': 15
    }
    with open(topic_path / "qa_runtime_stats.json", 'w') as f:
        json.dump(qa_stats, f)

    # Create mock SQuAD metrics
    squad_metrics = {
        'exact_match': 0.75,
        'f1': 0.82,
        'no_answer_probability': 0.15
    }
    with open(topic_path / "metrics.json", 'w') as f:
        json.dump(squad_metrics, f)

    # Create mock Neo4j metrics
    neo4j_path = topic_path / "neo4j_eval"
    neo4j_path.mkdir(exist_ok=True)

    neo4j_metrics = {
        'successful_queries': 12,
        'total_queries': 15,
        'success_rate': 0.8,
        'queries_with_results': 10,
        'results_rate': 0.67,
        'correct_queries': 8,
        'accuracy': 0.53,
        'total_elapsed_time_seconds': 25.3,
        'estimated_cost_usd': 0.045
    }
    with open(neo4j_path / "neo4j_metrics.json", 'w') as f:
        json.dump(neo4j_metrics, f)

    # Create mock Neo4j runtime stats CSV
    import pandas as pd
    neo4j_runtime_data = [
        {'metric': 'total_elapsed_time_seconds', 'value': 25.3, 'formatted_value': '25.30'},
        {'metric': 'total_prompt_tokens', 'value': 1500, 'formatted_value': '1,500'},
        {'metric': 'total_completion_tokens', 'value': 800, 'formatted_value': '800'},
        {'metric': 'total_reasoning_tokens', 'value': 0, 'formatted_value': '0'},
        {'metric': 'total_cached_tokens', 'value': 200, 'formatted_value': '200'},
        {'metric': 'total_tokens', 'value': 2300, 'formatted_value': '2,300'},
        {'metric': 'total_calls', 'value': 12, 'formatted_value': '12'},
        {'metric': 'estimated_cost_usd', 'value': 0.045, 'formatted_value': '$0.0450'}
    ]
    df = pd.DataFrame(neo4j_runtime_data)
    df.to_csv(neo4j_path / "neo4j_runtime_stats.csv", index=False)

def test_run_metrics():
    """Test the run metrics generation functionality."""

    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)

        # Create mock evaluation data
        create_mock_evaluation_data(base_path)

        # Create mock config
        config = type('MockConfig', (), {
            'scenarios': [
                EvalScenario(
                    id="biomedical_engineer",
                    domain="biomedical engineering",
                    squad_titles=("Biomedical Engineer",),
                    neo4j=Neo4jConfig(enabled=True),
                    dataset_mode="test"
                )
            ]
        })()

        # Test the metrics generation
        try:
            run_metrics = _generate_run_metrics_and_stats(base_path, config)

            print("✅ Run metrics generation successful!")
            print(f"Total scenarios: {run_metrics['total_scenarios']}")
            print(f"Total topics: {run_metrics['total_topics']}")
            print(f"Scenarios with ontology: {run_metrics['scenarios_with_ontology']}")
            print(f"Neo4j enabled scenarios: {run_metrics['neo4j_enabled_scenarios']}")

            if 'overall_squad_metrics' in run_metrics:
                print(f"Overall SQuAD F1: {run_metrics['overall_squad_metrics']['f1']:.4f}")

            if 'overall_neo4j_metrics' in run_metrics:
                print(f"Overall Neo4j success rate: {run_metrics['overall_neo4j_metrics']['success_rate']:.4f}")

            print(f"Total runtime: {run_metrics['runtime_stats']['total_elapsed_time_seconds']:.2f} seconds")
            print(f"Total tokens: {run_metrics['runtime_stats']['total_tokens']:,}")
            print(f"Estimated cost: ${run_metrics['runtime_stats']['estimated_cost_usd']:.4f}")

            # Check if output files were created
            run_metrics_file = base_path / "run_metrics.json"
            runtime_stats_file = base_path / "run_runtime_stats.csv"

            if run_metrics_file.exists():
                print("✅ run_metrics.json created successfully")
            else:
                print("❌ run_metrics.json not found")

            if runtime_stats_file.exists():
                print("✅ run_runtime_stats.csv created successfully")
            else:
                print("❌ run_runtime_stats.csv not found")

        except Exception as e:
            print(f"❌ Error during metrics generation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_run_metrics()