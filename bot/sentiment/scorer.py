"""
bot/sentiment/scorer.py
------------------------
Scores raw text using FinBERT (finance-specific BERT model).
Falls back to VADER if FinBERT model is not available.

FinBERT scores each piece of text as:
  positive / negative / neutral
and returns a single float score from -1.0 (bearish) to +1.0 (bullish).
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"


class SentimentScorer:
    """
    Scores financial text using FinBERT with VADER fallback.

    Usage:
        scorer = SentimentScorer()
        score = scorer.score("Apple reported record earnings today")
        # Returns float between -1.0 and +1.0
    """

    def __init__(self, backend: Literal["finbert", "vader", "auto"] = "auto"):
        self.backend = backend
        self._pipeline = None
        self._vader    = None
        self._init_backend()

    def _init_backend(self):
        if self.backend in ("finbert", "auto"):
            try:
                # Import the pipeline factory from the submodule rather
                # than the top-level package.  transformers >= 4.57 uses a
                # lazy ``__getattr__`` resolver on its ``__init__.py`` that
                # can fail with "cannot import name 'pipeline' from
                # 'transformers'" when something else has already touched a
                # transformers submodule (sentiment_pipeline does this when
                # it pre-imports the news fetcher).  Hitting the submodule
                # path bypasses the resolver entirely and is the official
                # workaround.
                from transformers.pipelines import pipeline
                self._pipeline = pipeline(
                    "text-classification",
                    model=FINBERT_MODEL,
                    tokenizer=FINBERT_MODEL,
                    top_k=None,
                )
                self.backend = "finbert"
                logger.info("SentimentScorer using FinBERT.")
                return
            except Exception as exc:
                logger.warning("FinBERT unavailable (%s) — falling back to VADER.", exc)

        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            self._vader   = SentimentIntensityAnalyzer()
            self.backend  = "vader"
            logger.info("SentimentScorer using VADER.")
        except Exception as exc:
            logger.error("Neither FinBERT nor VADER available: %s", exc)
            self.backend = "none"

    def score(self, text: str) -> float:
        """
        Score a single piece of text.

        Args:
            text: Headline, summary, or post body.

        Returns:
            Float in [-1.0, +1.0]. Positive = bullish, negative = bearish.
        """
        if not text or not text.strip():
            return 0.0

        text = text[:512]

        if self.backend == "finbert" and self._pipeline:
            return self._score_finbert(text)
        if self.backend == "vader" and self._vader:
            return self._score_vader(text)
        return 0.0

    def score_batch(self, texts: list[str]) -> list[float]:
        """
        Score a list of texts efficiently.

        Args:
            texts: List of text strings.

        Returns:
            List of floats in [-1.0, +1.0].
        """
        if not texts:
            return []
        if self.backend == "finbert" and self._pipeline:
            return [self._score_finbert(t) for t in texts]
        return [self.score(t) for t in texts]

    def _score_finbert(self, text: str) -> float:
        """Convert FinBERT label probabilities to a [-1, +1] score."""
        try:
            results = self._pipeline(text)[0]
            scores  = {r["label"].lower(): r["score"] for r in results}
            return float(scores.get("positive", 0) - scores.get("negative", 0))
        except Exception as exc:
            logger.warning("FinBERT scoring error: %s", exc)
            return 0.0

    def _score_vader(self, text: str) -> float:
        """Use VADER compound score as a proxy (-1 to +1)."""
        try:
            return float(self._vader.polarity_scores(text)["compound"])
        except Exception as exc:
            logger.warning("VADER scoring error: %s", exc)
            return 0.0
