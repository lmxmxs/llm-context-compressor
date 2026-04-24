"""
Tests for context_compressor.py
Run: python -m pytest tests/test_context_compressor.py -v
"""
import sys
import json
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from context_compressor import (
    compress_qa_pairs,
    compress_sources,
    smart_truncate_context,
    estimate_compression_ratio,
    _density_score,
    _extract_topic_words,
)


# ── _density_score ────────────────────────────────────────────────────────────

class TestDensityScore:
    def test_empty_returns_zero(self):
        assert _density_score("") == 0.0
        assert _density_score("ab") == 0.0

    def test_numeric_density_boost(self):
        factual = "The experiment ran at 98.6°F for 42 minutes yielding 73% efficiency."
        generic = "The experiment was done and it worked well in the end."
        assert _density_score(factual) > _density_score(generic)

    def test_topic_overlap_boost(self):
        topic = {"quantum", "computing", "qubit", "superposition"}
        relevant = "Quantum computing uses qubits for superposition states."
        irrelevant = "The weather was nice today and the birds were singing."
        assert _density_score(relevant, topic) > _density_score(irrelevant, topic)

    def test_very_long_sentence_penalized(self):
        short = "Photosynthesis converts sunlight into glucose using chlorophyll."
        long = " ".join(["word"] * 200)  # 200 identical words — low unique ratio + length penalty
        assert _density_score(short) > _density_score(long)


# ── _extract_topic_words ──────────────────────────────────────────────────────

class TestExtractTopicWords:
    def test_basic(self):
        words = _extract_topic_words("machine learning algorithms")
        assert "machine" in words
        assert "learning" in words
        assert "algorithms" in words

    def test_stopwords_excluded(self):
        words = _extract_topic_words("how does this work for the user")
        assert "does" not in words
        assert "this" not in words
        assert "for" not in words

    def test_with_questions(self):
        words = _extract_topic_words("neural networks", ["What is backpropagation?", "How does gradient descent work?"])
        assert "backpropagation" in words
        assert "gradient" in words
        assert "descent" in words


# ── compress_qa_pairs ─────────────────────────────────────────────────────────

class TestCompressQaPairs:
    def _sample_qa(self, n=5):
        return [
            {"question": f"Question {i} about the topic?",
             "answer": f"Answer {i}: " + "Detail. " * (10 + i)}
            for i in range(n)
        ]

    def test_empty_returns_empty(self):
        assert compress_qa_pairs([]) == ""

    def test_output_within_max_chars(self):
        qa = self._sample_qa(10)
        result = compress_qa_pairs(qa, max_chars=500)
        assert len(result) <= 520  # small tolerance for last pair truncation marker

    def test_output_contains_qa_format(self):
        qa = [{"question": "What is X?", "answer": "X is a thing."}]
        result = compress_qa_pairs(qa, max_chars=200)
        assert "P:" in result
        assert "O:" in result

    def test_custom_keys(self):
        qa = [{"q": "What?", "a": "This."}]
        result = compress_qa_pairs(qa, max_chars=200, question_key="q", answer_key="a")
        assert "What?" in result
        assert "This." in result

    def test_informative_pairs_prioritized(self):
        qa = [
            {"question": "Basic?", "answer": "Yes it is."},
            {"question": "Stats?", "answer": "The study found 87.3% accuracy in 1,240 trials with p<0.001, yielding 3.2x improvement."},
        ]
        result = compress_qa_pairs(qa, max_chars=120, topic="statistics")
        assert "87.3%" in result or "1,240" in result

    def test_long_answer_truncated(self):
        qa = [{"question": "Q?", "answer": "A" * 5000}]
        result = compress_qa_pairs(qa, max_chars=2000)
        assert len(result) <= 2050

    def test_many_pairs_within_budget(self):
        qa = self._sample_qa(50)
        result = compress_qa_pairs(qa, max_chars=2000)
        assert len(result) <= 2050
        assert "P:" in result


# ── compress_sources ──────────────────────────────────────────────────────────

class TestCompressSources:
    def _sample_sources(self, n=5):
        return [
            {"title": f"Article {i}", "url": f"https://example.com/{i}"}
            for i in range(n)
        ]

    def test_empty_returns_marker(self):
        result = compress_sources([])
        assert result == "(brak źródeł)"

    def test_max_items_respected(self):
        sources = self._sample_sources(30)
        result = compress_sources(sources, max_items=5)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) <= 5

    def test_numbering(self):
        sources = self._sample_sources(3)
        result = compress_sources(sources)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_include_content(self):
        sources = [{"title": "Test", "url": "https://x.com", "content": "Full content here " * 50}]
        result = compress_sources(sources, include_content=True, max_content_chars=50)
        assert "→" in result
        assert len([l for l in result.split("\n") if "→" in l][0]) <= 200


# ── smart_truncate_context ────────────────────────────────────────────────────

class TestSmartTruncateContext:
    def test_short_text_unchanged(self):
        text = "Short text that fits easily."
        assert smart_truncate_context(text, max_chars=1000) == text

    def test_output_within_max(self):
        text = "Sentence number {}. ".format(1) * 200
        result = smart_truncate_context(text, max_chars=500)
        assert len(result) <= 510

    def test_front_preserved(self):
        sentences = [f"Sentence {i}." for i in range(100)]
        text = " ".join(sentences)
        result = smart_truncate_context(text, max_chars=200)
        assert "Sentence 0" in result

    def test_ellipsis_marker(self):
        text = " ".join([f"Sentence {i} with some words." for i in range(100)])
        result = smart_truncate_context(text, max_chars=300)
        # either truncated at max_chars or contains the ellipsis separator
        assert len(result) <= 310 or " [...] " in result


# ── estimate_compression_ratio ────────────────────────────────────────────────

class TestEstimateCompressionRatio:
    def test_identical_is_one(self):
        assert estimate_compression_ratio("abc", "abc") == 1.0

    def test_empty_original(self):
        assert estimate_compression_ratio("", "anything") == 1.0

    def test_half_compression(self):
        ratio = estimate_compression_ratio("a" * 100, "a" * 50)
        assert abs(ratio - 0.5) < 0.01

    def test_lower_is_better(self):
        ratio_good = estimate_compression_ratio("a" * 1000, "a" * 100)
        ratio_bad = estimate_compression_ratio("a" * 1000, "a" * 900)
        assert ratio_good < ratio_bad


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, args, stdin=None):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parents[1] / "context_compressor.py")] + args,
            input=stdin, capture_output=True, text=True
        )
        return result

    def test_help(self):
        r = self._run(["--help"])
        assert r.returncode == 0
        assert "qa" in r.stdout

    def test_qa_from_stdin(self):
        data = json.dumps([{"question": "What?", "answer": "This is it."}])
        r = self._run(["qa", "-", "--max-chars", "200"], stdin=data)
        assert r.returncode == 0
        assert "What?" in r.stdout

    def test_truncate_from_stdin(self):
        text = "Word " * 500
        r = self._run(["truncate", "-", "--max-chars", "100"], stdin=text)
        assert r.returncode == 0
        assert len(r.stdout.strip()) <= 120

    def test_stats_flag(self):
        data = json.dumps([{"question": "Q?", "answer": "A. " * 100}])
        r = self._run(["qa", "-", "--max-chars", "100", "--stats"], stdin=data)
        assert "stats:" in r.stderr
