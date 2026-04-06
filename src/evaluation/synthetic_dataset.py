"""
Synthetic evaluation dataset generator.

Uses Claude to create 200 Q&A pairs from sampled SEC filing chunks,
covering 4 question categories:
  1. Simple factual (50 pairs)
  2. Numerical reasoning (50 pairs)
  3. Multi-document comparison (25 pairs)
  4. Temporal / year-over-year trends (25 pairs)
  5. Risk factors (25 pairs)
  6. Business description (25 pairs)

Output: data/eval_dataset.json
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

QA_GENERATION_PROMPT = """\
You are a financial QA dataset generator.

Below is an excerpt from a SEC filing ({company} — {form_type}, {filing_date}, {section}).

Generate {n} high-quality question-answer pairs from this excerpt.
Requirements:
- Questions must be answerable from the excerpt alone
- Answers must be factual and specific (include numbers, dates, percentages where present)
- Question types should be: {question_type}
- Format: JSON array of {{"question": "...", "answer": "...", "ground_truth_context": "..."}}

Excerpt:
{excerpt}

Output only the JSON array, no explanation."""

QUESTION_TYPES = {
    "factual": "specific factual questions about financial figures, dates, or named entities",
    "numerical": "questions requiring arithmetic or percentage calculations from the data",
    "temporal": "questions about year-over-year changes, trends, or time-series data",
    "risk": "questions about risk factors, business challenges, or uncertainties",
    "business": "questions about business model, operations, or strategic initiatives",
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SyntheticDatasetGenerator:
    """Generates a balanced synthetic evaluation dataset from indexed chunks."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate(
        self,
        num_samples: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> list[dict]:
        """
        Generate synthetic QA pairs.

        Args:
            num_samples: total number of QA pairs (default: settings.eval_sample_size)
            output_path: where to save JSON (default: settings.eval_dataset_path)

        Returns:
            List of QA pair dicts
        """
        num_samples = num_samples or settings.eval_sample_size
        output_path = output_path or settings.eval_dataset_path

        # Load sample chunks from processed documents
        chunks = self._sample_chunks(num_samples * 2)  # oversample to account for failures
        if not chunks:
            raise RuntimeError("No processed documents found. Run ingestion pipeline first.")

        # Distribute question types
        type_distribution = self._build_type_distribution(num_samples)

        all_pairs: list[dict] = []
        for q_type, count in type_distribution.items():
            logger.info("Generating %d '%s' QA pairs", count, q_type)
            type_chunks = random.sample(chunks, min(count * 2, len(chunks)))

            pairs = self._generate_pairs(type_chunks, q_type, target_count=count)
            all_pairs.extend(pairs)

            if len(all_pairs) >= num_samples:
                break

        all_pairs = all_pairs[:num_samples]
        logger.info("Generated %d QA pairs", len(all_pairs))

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_pairs, indent=2, ensure_ascii=False))
        logger.info("Saved evaluation dataset to %s", output_path)

        return all_pairs

    def _build_type_distribution(self, total: int) -> dict[str, int]:
        """Distribute QA pairs across question types."""
        return {
            "factual": int(total * 0.30),
            "numerical": int(total * 0.25),
            "temporal": int(total * 0.20),
            "risk": int(total * 0.15),
            "business": total - int(total * 0.30) - int(total * 0.25) - int(total * 0.20) - int(total * 0.15),
        }

    def _sample_chunks(self, n: int):
        """Sample n chunks from processed documents."""
        from src.ingestion.parser import iter_parsed_documents
        from src.ingestion.chunker import FilingChunker

        chunker = FilingChunker()
        all_chunks = []
        for doc in iter_parsed_documents(settings.processed_data_dir):
            chunks = chunker.chunk(doc)
            # Prefer text chunks with substantial content for QA generation
            rich_chunks = [c for c in chunks if c.chunk_type == "text" and len(c.text) > 200]
            all_chunks.extend(rich_chunks)
            if len(all_chunks) >= n * 3:
                break

        if not all_chunks:
            return []

        return random.sample(all_chunks, min(n, len(all_chunks)))

    def _generate_pairs(self, chunks, q_type: str, target_count: int) -> list[dict]:
        """Generate QA pairs of a specific type from chunks."""
        pairs: list[dict] = []
        pairs_per_chunk = max(1, target_count // max(len(chunks), 1))

        for chunk in chunks:
            if len(pairs) >= target_count:
                break
            if len(chunk.text.strip()) < 100:
                continue

            prompt = QA_GENERATION_PROMPT.format(
                company=chunk.company_name,
                form_type=chunk.form_type,
                filing_date=chunk.filing_date,
                section=chunk.section,
                n=min(pairs_per_chunk + 1, 3),
                question_type=QUESTION_TYPES[q_type],
                excerpt=chunk.text[:1500],
            )

            try:
                message = self._client.messages.create(
                    model=settings.claude_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text.strip()

                # Parse JSON array
                import re
                json_match = re.search(r'\[.*\]', raw, re.DOTALL)
                if not json_match:
                    continue
                generated = json.loads(json_match.group())

                for item in generated:
                    if "question" in item and "answer" in item:
                        pairs.append({
                            "question": item["question"],
                            "answer": item["answer"],
                            "ground_truth": item.get("ground_truth_context", item["answer"]),
                            "context": chunk.text,
                            "metadata": chunk.metadata(),
                            "question_type": q_type,
                        })

            except Exception as exc:
                logger.debug("QA generation failed for chunk: %s", exc)
                continue

        return pairs


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load_eval_dataset(path: Optional[Path] = None) -> list[dict]:
    """Load the evaluation dataset from JSON."""
    path = path or settings.eval_dataset_path
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {path}. "
            "Run: python scripts/run_evaluation.py --generate-dataset"
        )
    return json.loads(path.read_text())
