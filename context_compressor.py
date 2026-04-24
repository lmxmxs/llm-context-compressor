"""
context_compressor — Lightweight LLM context compression without dependencies.

Compresses Q&A pairs, source lists, and long documents before sending to LLM APIs.
Uses information density scoring inspired by LLMLingua / LongLLMLingua research,
but implemented with pure Python — no transformers, no PyTorch, no GPU required.

Key functions:
    compress_qa_pairs()      — compress list of Q&A dicts to max_chars
    compress_sources()       — format source list for LLM context
    smart_truncate_context() — LongLLMLingua-inspired: keep front+important_middle+back

Typical usage:
    from context_compressor import compress_qa_pairs
    compressed = compress_qa_pairs(qa_list, max_chars=6000, topic="quantum computing")
    # Feed compressed into your LLM prompt instead of raw qa_list

CLI usage:
    python context_compressor.py --help
    python context_compressor.py qa input.json --max-chars 4000 --topic "AI safety"
    python context_compressor.py truncate long_text.txt --max-chars 2000

License: MIT
"""
import re
import math
import json
import sys
import argparse
from typing import List, Dict, Optional, Tuple


def _density_score(text: str, topic_words: set = None) -> float:
    """
    Oblicza gęstość informacyjną zdania/akapitu.
    Wyższy wynik = więcej unikalnej, istotnej informacji.
    """
    if not text or len(text) < 20:
        return 0.0

    words = re.findall(r'\b\w{3,}\b', text.lower())
    if not words:
        return 0.0

    # Unikalne słowa / total → miara nowości
    unique_ratio = len(set(words)) / len(words)

    # Liczby, daty, jednostki → sygnał faktograficzny
    numeric_density = len(re.findall(r'\d+[,.]?\d*\s*(%|mg|kg|km|ml|°|USD|PLN|B|M|K)?', text)) / max(len(words), 1)

    # Długość zdania: preferuj 15-40 słów (nie za krótkie, nie za długie)
    word_count = len(words)
    length_score = 1.0 if 15 <= word_count <= 80 else (word_count / 15 if word_count < 15 else 80 / word_count)

    # Overlap z tematem (pytaniem / słowami kluczowymi)
    topic_score = 0.0
    if topic_words:
        overlap = len(set(words) & topic_words)
        topic_score = min(overlap / max(len(topic_words), 1), 1.0) * 0.3

    return (unique_ratio * 0.4 + numeric_density * 2.0 + length_score * 0.3 + topic_score)


def _extract_topic_words(topic: str, questions: List[str] = None) -> set:
    """Wyodrębnij słowa kluczowe tematu i pytań badawczych."""
    text = (topic or "") + " " + " ".join(questions or [])
    stopwords = {
        "jest", "są", "jak", "co", "czy", "dla", "przez", "oraz",
        "które", "tego", "jego", "jej", "ich", "jako", "przy", "więc",
        "the", "and", "for", "that", "this", "with", "have", "from",
        "does", "what", "when", "where", "which", "they", "them", "their",
        "been", "will", "would", "could", "should", "about", "more", "some",
        "work", "make", "just", "also", "into", "very", "over",
    }
    words = {w for w in re.findall(r'\b\w{4,}\b', text.lower()) if w not in stopwords}
    return words


def compress_qa_pairs(
    qa_list: List[Dict],
    max_chars: int = 8000,
    topic: str = "",
    questions: List[str] = None,
    question_key: str = "question",
    answer_key: str = "answer",
) -> str:
    """
    Kompresuje listę Q&A par do max_chars znaków.

    Algorytm:
    1. Score każdej pary (density odpowiedzi + overlap z tematem)
    2. Sortuj malejąco po score
    3. Buduj tekst: bierz po kolei aż do limitu
    4. Fallback: truncate ostatnią parę

    Returns: sformatowany tekst Q&A do wstrzyknięcia w prompt
    """
    if not qa_list:
        return ""

    topic_words = _extract_topic_words(topic, questions)

    # Oceń każdą parę
    scored = []
    for i, item in enumerate(qa_list):
        q = str(item.get(question_key) or item.get("q") or "")
        a = str(item.get(answer_key) or item.get("a") or "")
        if not q and not a:
            continue
        score = _density_score(a, topic_words)
        # Boost dla pierwszych par (często zawierają definicje/kontekst)
        if i < 3:
            score *= 1.2
        scored.append((score, i, q, a))

    # Sortuj: wysoki score first, ale zachowaj mix (nie tylko top)
    scored.sort(key=lambda x: -x[0])

    # Buduj tekst
    result_parts = []
    total_chars = 0

    for score, original_idx, q, a in scored:
        # Truncate odpowiedź jeśli za długa
        a_trimmed = a[:1200] if len(a) > 1200 else a
        pair_text = f"P: {q}\nO: {a_trimmed}"
        pair_len = len(pair_text) + 2  # +2 dla \n\n

        if total_chars + pair_len > max_chars:
            remaining = max_chars - total_chars - 20
            if remaining > 100 and result_parts:
                a_short = a[:remaining]
                result_parts.append(f"P: {q}\nO: {a_short}...")
            break

        result_parts.append(pair_text)
        total_chars += pair_len

    return "\n\n".join(result_parts)


def compress_sources(
    sources: List[Dict],
    max_items: int = 20,
    max_content_chars: int = 300,
    include_content: bool = False,
) -> str:
    """
    Kompresuje listę źródeł do czytelnej formy dla modelu.
    Opcjonalnie dołącza skrócony content (dla modeli bez RAG).
    """
    if not sources:
        return "(brak źródeł)"

    lines = []
    for i, s in enumerate(sources[:max_items]):
        title = s.get("title") or s.get("url") or f"Źródło {i+1}"
        url = s.get("url", "")
        line = f"[{i+1}] {title}"
        if url and url not in title:
            line += f": {url}"

        if include_content:
            content = s.get("content") or s.get("snippet") or ""
            if content:
                content_short = content[:max_content_chars].replace("\n", " ")
                line += f"\n    → {content_short}"

        lines.append(line)

    return "\n".join(lines)


def smart_truncate_context(
    text: str,
    max_chars: int,
    topic: str = "",
) -> str:
    """
    LongLLMLingua-inspired: inteligentne skracanie długiego tekstu.
    Zachowuje: pierwsze 40% + najważniejsze zdania ze środka + ostatnie 20%.
    Usuwa: mało informacyjne zdania ze środka.
    """
    if len(text) <= max_chars:
        return text

    topic_words = _extract_topic_words(topic)

    # Podziel na zdania
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 3:
        return text[:max_chars]

    keep_front = int(len(sentences) * 0.35)
    keep_back = int(len(sentences) * 0.15)
    middle = sentences[keep_front:-keep_back] if keep_back > 0 else sentences[keep_front:]

    front_text = " ".join(sentences[:keep_front])
    back_text = " ".join(sentences[-keep_back:]) if keep_back > 0 else ""

    # Ile znaków zostało na środek
    front_back_len = len(front_text) + len(back_text) + 10
    middle_budget = max_chars - front_back_len

    # Wybierz najlepsze zdania ze środka
    if middle_budget > 100 and middle:
        scored_middle = [
            (_density_score(s, topic_words), i, s)
            for i, s in enumerate(middle)
            if len(s) > 20
        ]
        scored_middle.sort(key=lambda x: -x[0])

        middle_parts = []
        middle_chars = 0
        for _, _, s in scored_middle:
            if middle_chars + len(s) + 1 > middle_budget:
                break
            middle_parts.append(s)
            middle_chars += len(s) + 1

        middle_text = " [...] ".join(middle_parts) if middle_parts else ""
    else:
        middle_text = ""

    parts = [p for p in [front_text, middle_text, back_text] if p]
    result = " [...] ".join(parts)
    return result[:max_chars]


def estimate_compression_ratio(original: str, compressed: str) -> float:
    """Return compression ratio (0-1, lower = better compression)."""
    if not original:
        return 1.0
    return len(compressed) / len(original)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="context_compressor — lightweight LLM context compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python context_compressor.py qa input.json --max-chars 4000 --topic "AI safety"
  python context_compressor.py truncate long_text.txt --max-chars 2000 --topic "quantum"
  cat qa_data.json | python context_compressor.py qa - --max-chars 5000
        """
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # qa: kompresja Q&A par
    p_qa = sub.add_parser("qa", help="Compress Q&A pairs from JSON file")
    p_qa.add_argument("input", help="JSON file with Q&A list (or - for stdin)")
    p_qa.add_argument("--max-chars", type=int, default=6000, help="Max output chars (default: 6000)")
    p_qa.add_argument("--topic", default="", help="Topic for relevance scoring")
    p_qa.add_argument("--question-key", default="question", help="JSON key for question field")
    p_qa.add_argument("--answer-key", default="answer", help="JSON key for answer field")
    p_qa.add_argument("--stats", action="store_true", help="Show compression stats")

    # truncate: smart truncation długiego tekstu
    p_trunc = sub.add_parser("truncate", help="Smart-truncate long text (front+middle+back)")
    p_trunc.add_argument("input", help="Text file (or - for stdin)")
    p_trunc.add_argument("--max-chars", type=int, default=4000, help="Max output chars (default: 4000)")
    p_trunc.add_argument("--topic", default="", help="Topic for middle sentence selection")
    p_trunc.add_argument("--stats", action="store_true", help="Show compression stats")

    args = parser.parse_args()

    if args.cmd == "qa":
        src = sys.stdin.read() if args.input == "-" else open(args.input).read()
        qa_list = json.loads(src)
        if not isinstance(qa_list, list):
            print("Error: input must be a JSON array of Q&A objects", file=sys.stderr)
            sys.exit(1)
        result = compress_qa_pairs(
            qa_list,
            max_chars=args.max_chars,
            topic=args.topic,
            question_key=args.question_key,
            answer_key=args.answer_key,
        )
        print(result)
        if args.stats:
            original = "\n\n".join(
                f"Q: {a.get(args.question_key,'')}\nA: {a.get(args.answer_key,'')}"
                for a in qa_list
            )
            ratio = estimate_compression_ratio(original, result)
            print(f"\n--- stats: {len(original)} → {len(result)} chars ({ratio:.1%} of original) ---",
                  file=sys.stderr)

    elif args.cmd == "truncate":
        src = sys.stdin.read() if args.input == "-" else open(args.input).read()
        result = smart_truncate_context(src, max_chars=args.max_chars, topic=args.topic)
        print(result)
        if args.stats:
            ratio = estimate_compression_ratio(src, result)
            print(f"\n--- stats: {len(src)} → {len(result)} chars ({ratio:.1%} of original) ---",
                  file=sys.stderr)


if __name__ == "__main__":
    _cli()
