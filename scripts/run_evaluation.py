"""
CLI script: Run RAG evaluation (RAGAS and/or DeepEval).

Examples:
    # Generate synthetic dataset + run both evaluators
    python scripts/run_evaluation.py --all

    # Generate 200 synthetic QA pairs only
    python scripts/run_evaluation.py --generate-dataset

    # Run RAGAS on 50 samples
    python scripts/run_evaluation.py --ragas --sample 50

    # Run DeepEval on 30 samples
    python scripts/run_evaluation.py --deepeval --sample 30

    # Run both evaluators
    python scripts/run_evaluation.py --ragas --deepeval

Results saved to:
    data/eval_dataset.json
    data/ragas_results.csv
    data/deepeval_results.json
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--generate-dataset", action="store_true",
        help="Generate synthetic QA dataset before evaluation",
    )
    parser.add_argument(
        "--num-samples", type=int, default=settings.eval_sample_size,
        help=f"Number of QA pairs to generate (default: {settings.eval_sample_size})",
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Run RAGAS evaluation",
    )
    parser.add_argument(
        "--deepeval", action="store_true",
        help="Run DeepEval evaluation",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate dataset + run both RAGAS and DeepEval",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit number of eval items (speeds up evaluation)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        args.generate_dataset = True
        args.ragas = True
        args.deepeval = True

    # Generate dataset
    if args.generate_dataset:
        logger.info("Generating synthetic evaluation dataset (%d pairs)", args.num_samples)
        from src.evaluation.synthetic_dataset import SyntheticDatasetGenerator
        gen = SyntheticDatasetGenerator()
        pairs = gen.generate(num_samples=args.num_samples)
        print(f"Generated {len(pairs)} QA pairs → {settings.eval_dataset_path}")

    # RAGAS evaluation
    if args.ragas:
        logger.info("Running RAGAS evaluation (sample=%s)", args.sample)
        from src.evaluation.ragas_eval import RAGASEvaluator
        scores = RAGASEvaluator().evaluate(sample_size=args.sample)

        print("\n=== RAGAS Results ===")
        for metric, score in scores.items():
            threshold = getattr(settings, f"ragas_{metric}_threshold", None)
            if isinstance(score, float):
                status = ""
                if threshold is not None:
                    status = " ✓ PASS" if score >= threshold else " ✗ FAIL"
                print(f"  {metric:30s}: {score:.4f}{status}")

    # DeepEval evaluation
    if args.deepeval:
        logger.info("Running DeepEval evaluation (sample=%s)", args.sample)
        from src.evaluation.deepeval_eval import DeepEvalEvaluator
        scores = DeepEvalEvaluator().evaluate(sample_size=args.sample)

        print("\n=== DeepEval Results ===")
        for metric, data in scores.items():
            pass_rate = data.get("pass_rate", 0)
            print(f"  {metric:35s}: {pass_rate:.1%} pass rate")

    if not any([args.generate_dataset, args.ragas, args.deepeval]):
        print("No action specified. Use --all, --ragas, --deepeval, or --generate-dataset.")
        print("Run with --help for usage.")


if __name__ == "__main__":
    main()
