import json
import logging
from pathlib import Path
from typing import List, Dict
from statistics import mean

from automa_ai.mcp_servers.agent_card_server import build_agent_card_embeddings, find_best_match

logger = logging.getLogger(__name__)


def load_test_cases(test_file: str) -> List[Dict]:
    """
    Load test cases from a JSON file.
    Each entry should look like:
    {
        "query": "Find the agent that helps with EnergyPlus schema validation",
        "expected_uri": "resource://agent_cards/energyplus_agent"
    }
    """
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    with path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    logger.info(f"Loaded {len(cases)} test cases.")
    return cases


def evaluate_agent_card_matching(
    agent_card_dir: str,
    test_file: str,
    persist_dir: str = "./chroma_store",
    verbose: bool = True,
):
    """
    Evaluate how accurately find_best_match() identifies the correct agent.
    """
    # Step 1: Rebuild index (fresh start)
    build_agent_card_embeddings(agent_card_dir, persist_dir=persist_dir)

    # Step 2: Load test cases
    test_cases = load_test_cases(test_file)

    total = len(test_cases)
    correct = 0
    scores = []
    mismatches = []

    # Step 3: Evaluate
    for case in test_cases:
        query = case["query"]
        expected_uri = case["expected_uri"]

        result = find_best_match(query, persist_dir=persist_dir)
        if not result:
            mismatches.append((query, expected_uri, None))
            continue

        predicted_uri = result["uri"]
        distance = result["distance"]
        scores.append(distance)

        if predicted_uri == expected_uri:
            correct += 1
            if verbose:
                logger.info(f"✅ PASS: {query} → {predicted_uri}")
        else:
            mismatches.append((query, expected_uri, predicted_uri))
            if verbose:
                logger.warning(f"❌ FAIL: {query} → {predicted_uri} (expected {expected_uri})")

    accuracy = correct / total if total else 0
    avg_distance = mean(scores) if scores else float("nan")

    logger.info("==== Evaluation Summary ====")
    logger.info(f"Total tests: {total}")
    logger.info(f"Correct:     {correct}")
    logger.info(f"Accuracy:    {accuracy:.2%}")
    logger.info(f"Avg distance:{avg_distance:.4f}")
    if mismatches:
        logger.info(f"Mismatches ({len(mismatches)}):")
        for q, exp, pred in mismatches:
            logger.info(f"  Q: {q}\n  → expected: {exp}\n  → got: {pred}\n")

    return {
        "accuracy": accuracy,
        "avg_distance": avg_distance,
        "mismatches": mismatches,
        "total_cases": total,
    }


# results = evaluate_agent_card_matching(
#    agent_card_dir="/Users/xuwe123/github/BEM-AI/examples/sim_bem_network/agent_cards",
#    test_file="/Users/xuwe123/github/BEM-AI/automa_ai/mcp_servers/test_cases.json",
#    persist_dir="./chroma_store",
#)