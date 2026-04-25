# llm-context-compressor

Lightweight Q&A and RAG context compression for LLM APIs.  
**Zero dependencies** — pure Python stdlib. No transformers, no PyTorch, no GPU.

```bash
pip install llm-context-compressor
```

---

## The problem

You're building a RAG pipeline. You have 50 Q&A pairs from your knowledge base, a list of sources, and a long document. Feeding it all raw to GPT-4 / Claude costs 20k tokens per request.

[LLMLingua](https://github.com/microsoft/LLMLingua) solves this with perplexity-based pruning — but requires `transformers` + PyTorch (500 MB+). That's too heavy for a simple API server or a lambda function.

`llm-context-compressor` gives you **70–80% compression** using information density scoring inspired by LLMLingua — implemented in ~200 lines of pure Python.

---

## Install

```bash
pip install llm-context-compressor
```

Or copy the single file:

```bash
curl -O https://raw.githubusercontent.com/lmxmxs/llm-context-compressor/main/context_compressor.py
```

---

## Usage

### Compress Q&A pairs

```python
from context_compressor import compress_qa_pairs

qa_list = [
    {"question": "What is RAG?", "answer": "Retrieval-Augmented Generation combines..."},
    {"question": "How does it work?", "answer": "It retrieves relevant documents..."},
    # ... 50 more pairs
]

compressed = compress_qa_pairs(qa_list, max_chars=6000, topic="retrieval augmented generation")
# Feed `compressed` into your LLM prompt instead of the raw list
```

**What it does:**
1. Scores each pair by information density (unique word ratio, numeric density, topic overlap)
2. Sorts by score — most informative pairs first
3. Fills the budget greedily, truncating the last pair if needed

### Compress source lists

```python
from context_compressor import compress_sources

sources = [
    {"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762"},
    {"title": "RAG paper", "url": "https://arxiv.org/abs/2005.11401", "content": "..."},
    # ...
]

formatted = compress_sources(sources, max_items=10, include_content=True, max_content_chars=200)
```

### Smart truncate long documents

```python
from context_compressor import smart_truncate_context

long_doc = open("document.txt").read()  # 50k chars

# LongLLMLingua-inspired: keeps front + high-density middle + back
truncated = smart_truncate_context(long_doc, max_chars=4000, topic="quantum computing")
```

### Measure compression

```python
from context_compressor import estimate_compression_ratio

ratio = estimate_compression_ratio(original_text, compressed_text)
print(f"Compression: {ratio:.1%} of original")  # e.g. "Compression: 23.4% of original"
```

---

## GUI (dark theme)

A standalone desktop interface — no server, no config, just run:

```bash
pip install PyQt6
python gui.py
```

Features:
- **Three modes**: Q&A Pairs · Smart Truncate · Sources
- **Compression ratio meter** — color-coded: green / amber / red
- **File open & save** — drag in any `.json` or `.txt`
- **Keyboard shortcuts**: `Ctrl+Enter` compress · `Ctrl+O` open · `Ctrl+S` save · `Ctrl+Shift+C` copy output

---

## CLI

```bash
# Compress Q&A pairs from JSON file
llm-compress qa data.json --max-chars 4000 --topic "machine learning" --stats

# Smart-truncate a long document
llm-compress truncate document.txt --max-chars 2000 --topic "quantum computing"

# Read from stdin
cat qa_data.json | llm-compress qa - --max-chars 5000

# Show compression stats
llm-compress qa data.json --max-chars 3000 --stats
# → --- stats: 18432 → 2987 chars (16.2% of original) ---
```

---

## How density scoring works

Each sentence / Q&A pair gets a score:

```
score = unique_word_ratio × 0.4
      + numeric_density   × 2.0   # numbers, dates, percentages = high signal
      + length_score      × 0.3   # penalizes very short or very long
      + topic_overlap     × 0.3   # overlap with your query/topic words
```

Pairs with facts ("87.3% accuracy in 1,240 trials, p<0.001") score higher than filler ("Yes, that is correct."). The budget is filled greedily from highest score downward.

---

## Benchmark

Tested on a 15-pair Q&A set from a RAG research pipeline (2.1k tokens raw):

| Method | Output tokens | Compression | Quality* |
|--------|---------------|-------------|---------|
| Raw (no compression) | 2,100 | 100% | baseline |
| Truncate first N | 840 | 40% | loses tail context |
| **llm-context-compressor** | **480** | **23%** | density-first |
| LLMLingua | 420 | 20% | PPL-based |

*Quality = subjective relevance of retained content for downstream LLM task.

LLMLingua achieves slightly better compression on average, but requires a full transformer model at inference time. `llm-context-compressor` is within 3% of LLMLingua's compression ratio at zero runtime overhead.

---

## API reference

```python
compress_qa_pairs(
    qa_list: list[dict],
    max_chars: int = 8000,
    topic: str = "",
    questions: list[str] = None,   # extra topic signal from research questions
    question_key: str = "question",
    answer_key: str = "answer",
) -> str

compress_sources(
    sources: list[dict],
    max_items: int = 20,
    max_content_chars: int = 300,
    include_content: bool = False,
) -> str

smart_truncate_context(
    text: str,
    max_chars: int,
    topic: str = "",
) -> str

estimate_compression_ratio(original: str, compressed: str) -> float
```

---

## Requirements

- Python 3.9+
- No external dependencies

---

## License

MIT
