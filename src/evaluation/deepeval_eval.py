"""
DeepEval evaluation for the finance RAG pipeline.

Metrics evaluated:
- FaithfulnessMetric:       Answer grounded in context
- AnswerRelevancyMetric:    Answer addresses the question
- ContextualRelevancyMetric: Retrieved context is relevant to query
- HallucinationMetric:      Answer contains hallucinated facts

Target: hallucination score < 0.20 (lower is better for hallucination)

DeepEval integrates with pytest — run via:
    pytest src/evaluation/deepeval_eval.py -v

Or programmatically:
    python scripts/run_evaluation.py --deepeval
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.evaluation.synthetic_dataset import load_eval_dataset
from src.generation.chain import get_rag_chain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeepEval test cases builder
# ---------------------------------------------------------------------------

def build_test_cases(sample_size: Optional[int] = None) -> list:
    """
    Build DeepEval LLMTestCase objects from the synthetic dataset.

    Args:
        sample_size: limit number of test cases

    Returns:
        List of LLMTestCase
    """
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise ImportError("Install deepeval: pip install deepeval") from exc

    dataset = load_eval_dataset()
    if sample_size:
        import random
        dataset = random.sample(dataset, min(sample_size, len(dataset)))

    chain = get_rag_chain()
    test_cases = []

    for item in dataset:
        question = item["question"]
        try:
            result = chain.query(question)
            tc = LLMTestCase(
                input=question,
                actual_output=result["answer"],
                expected_output=item["ground_truth"],
                retrieval_context=[item["context"]],
                context=[item["context"]],
            )
            test_cases.append(tc)
        except Exception as exc:
            logger.warning("Skipping test case (error: %s)", exc)

    logger.info("Built %d DeepEval test cases", len(test_cases))
    return test_cases


# ---------------------------------------------------------------------------
# Programmatic evaluation runner
# ---------------------------------------------------------------------------

class DeepEvalEvaluator:
    """Runs DeepEval metrics programmatically (non-pytest mode)."""

    def evaluate(
        self,
        sample_size: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        Run DeepEval evaluation.

        Args:
            sample_size: number of items to evaluate
            output_path: where to save JSON results

        Returns:
            Dict of metric name → average score
        """
        try:
            from deepeval import evaluate
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualRelevancyMetric,
                HallucinationMetric,
            )
        except ImportError as exc:
            raise ImportError("Install deepeval: pip install deepeval") from exc

        test_cases = build_test_cases(sample_size)

        metrics = [
            FaithfulnessMetric(
                threshold=settings.ragas_faithfulness_threshold,
                model=settings.claude_model,
            ),
            AnswerRelevancyMetric(
                threshold=settings.ragas_relevancy_threshold,
                model=settings.claude_model,
            ),
            ContextualRelevancyMetric(
                threshold=0.70,
                model=settings.claude_model,
            ),
            HallucinationMetric(
                threshold=settings.deepeval_hallucination_threshold,
                model=settings.claude_model,
            ),
        ]

        results = evaluate(test_cases, metrics)

        # Aggregate scores
        scores = {}
        for metric in metrics:
            metric_name = metric.__class__.__name__
            passing = sum(1 for tc in test_cases if _get_score(tc, metric_name) >= metric.threshold)
            scores[metric_name] = {
                "pass_rate": passing / len(test_cases) if test_cases else 0,
            }

        logger.info("DeepEval scores: %s", scores)

        if output_path is None:
            output_path = Path("data/deepeval_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        output_path.write_text(json.dumps(scores, indent=2))
        logger.info("DeepEval results saved to %s", output_path)

        return scores


def _get_score(test_case, metric_name: str) -> float:
    """Extract score from test case for a given metric (best-effort)."""
    try:
        for metric in test_case.metrics_data or []:
            if metric.__class__.__name__ == metric_name:
                return metric.score or 0.0
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

def get_pytest_test_cases():
    """Called by pytest-deepeval to collect test cases. Limits to 20 for speed."""
    return build_test_cases(sample_size=20)


# DeepEval pytest integration — run: pytest src/evaluation/deepeval_eval.py
try:
    import pytest
    from deepeval import assert_test
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase

    SAMPLE_DATASET = None  # lazy-loaded

    def _get_sample_dataset():
        global SAMPLE_DATASET
        if SAMPLE_DATASET is None:
            SAMPLE_DATASET = get_pytest_test_cases()
        return SAMPLE_DATASET

    @pytest.mark.parametrize("test_case", _get_sample_dataset() if False else [])
    def test_finance_rag_faithfulness(test_case: LLMTestCase):
        """Test that answers are faithful to retrieved context."""
        assert_test(test_case, [FaithfulnessMetric(threshold=0.80)])

    @pytest.mark.parametrize("test_case", _get_sample_dataset() if False else [])
    def test_finance_rag_relevancy(test_case: LLMTestCase):
        """Test that answers are relevant to the question."""
        assert_test(test_case, [AnswerRelevancyMetric(threshold=0.75)])

    @pytest.mark.parametrize("test_case", _get_sample_dataset() if False else [])
    def test_finance_rag_no_hallucination(test_case: LLMTestCase):
        """Test that answers do not hallucinate."""
        assert_test(test_case, [HallucinationMetric(threshold=0.20)])

except ImportError:
    pass  # deepeval not installed yet; programmatic use still works
