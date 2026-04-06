"""
RAGAS evaluation for the finance RAG pipeline.

Metrics evaluated:
- faithfulness:        Does the answer stick to the retrieved context?
- answer_relevancy:   Does the answer address the question?
- context_recall:     Does the context contain what's needed to answer?
- context_precision:  Is the context free of irrelevant chunks?

Target thresholds (from settings):
- faithfulness > 0.80
- answer_relevancy > 0.75

Usage::
    python scripts/run_evaluation.py --ragas [--sample 50]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import settings
from src.evaluation.synthetic_dataset import load_eval_dataset
from src.generation.chain import get_rag_chain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAGAS evaluation runner
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    Runs RAGAS evaluation over the synthetic dataset.
    Uses Claude as the judge LLM for RAGAS metrics.
    """

    def evaluate(
        self,
        sample_size: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        Run RAGAS evaluation.

        Args:
            sample_size: number of eval items to use (None = all)
            output_path: where to save CSV results

        Returns:
            Dict of metric name → score
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            )
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError("Install ragas: pip install ragas") from exc

        dataset = load_eval_dataset()
        if sample_size:
            import random
            dataset = random.sample(dataset, min(sample_size, len(dataset)))

        logger.info("Running RAGAS evaluation on %d items", len(dataset))

        # Build RAG outputs for each eval item
        chain = get_rag_chain()
        questions, answers, contexts, ground_truths = [], [], [], []

        for item in dataset:
            question = item["question"]
            try:
                result = chain.query(question)
                questions.append(question)
                answers.append(result["answer"])
                contexts.append([c["section"] + "\n" + item["context"] for c in result["sources"][:3]])
                ground_truths.append(item["ground_truth"])
            except Exception as exc:
                logger.warning("Skipping eval item (error: %s)", exc)

        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Run RAGAS — uses Claude as judge via LangChain wrapper
        from langchain_anthropic import ChatAnthropic
        from ragas.llms import LangchainLLMWrapper

        judge_llm = LangchainLLMWrapper(
            ChatAnthropic(
                model=settings.claude_model,
                anthropic_api_key=settings.anthropic_api_key,
            )
        )

        result = evaluate(
            eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=judge_llm,
        )

        scores = result.to_pandas().mean().to_dict()
        logger.info("RAGAS scores: %s", scores)

        # Save results
        if output_path is None:
            output_path = Path("data/ragas_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_pandas().to_csv(output_path, index=False)
        logger.info("RAGAS results saved to %s", output_path)

        # Log threshold checks
        self._check_thresholds(scores)
        return scores

    def _check_thresholds(self, scores: dict) -> None:
        """Log pass/fail for each threshold."""
        checks = {
            "faithfulness": settings.ragas_faithfulness_threshold,
            "answer_relevancy": settings.ragas_relevancy_threshold,
        }
        for metric, threshold in checks.items():
            score = scores.get(metric, 0.0)
            status = "PASS" if score >= threshold else "FAIL"
            logger.info(
                "[RAGAS] %s: %.3f (threshold=%.2f) → %s",
                metric, score, threshold, status,
            )
