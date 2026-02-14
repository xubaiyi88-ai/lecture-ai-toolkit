import sys
import re
import argparse
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_keywords(text: str, top_k: int = 10) -> list[str]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )
    tfidf = vectorizer.fit_transform([text])
    scores = tfidf.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    keywords = [term for term, score in ranked if score > 0][:top_k]
    return keywords


def summarize_text(text: str, num_sentences: int = 5) -> list[str]:
    text = clean_text(text)
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return sentences

    stop_words = set(stopwords.words("english"))

    word_freq: dict[str, int] = {}
    words = word_tokenize(text.lower())
    for w in words:
        if w.isalnum() and w not in stop_words:
            word_freq[w] = word_freq.get(w, 0) + 1

    if not word_freq:
        return sentences[:num_sentences]

    sentence_scores: list[tuple[int, float]] = []
    for idx, sent in enumerate(sentences):
        sent_words = word_tokenize(sent.lower())
        score = 0
        for w in sent_words:
            if w in word_freq:
                score += word_freq[w]
        sentence_scores.append((idx, float(score)))

    top = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_indices = sorted([idx for idx, _ in top])
    return [sentences[i] for i in top_indices]


def build_flashcards(text: str, keywords: list[str], max_cards: int = 10) -> list[tuple[str, str]]:
    sentences = sent_tokenize(clean_text(text))
    lower_sentences = [s.lower() for s in sentences]

    cards: list[tuple[str, str]] = []
    used = set()

    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in used:
            continue

        best_sentence = None
        for s, s_l in zip(sentences, lower_sentences):
            if kw_l in s_l:
                best_sentence = s
                break

        if best_sentence:
            cards.append((kw, best_sentence))
            used.add(kw_l)

        if len(cards) >= max_cards:
            break

    return cards


def generate_exam_questions(keywords: list[str], summary_sentences: list[str], max_q: int = 5) -> list[str]:
    questions: list[str] = []

    for kw in keywords[:3]:
        questions.append(f"Define '{kw}' and explain why it is important in this lecture topic.")

    for s in summary_sentences[:2]:
        questions.append(f"Explain the significance of the following idea and give an example: {s}")

    # If still short, add compare/apply prompts
    if len(questions) < max_q and len(keywords) >= 2:
        questions.append(f"Compare '{keywords[0]}' and '{keywords[1]}'. How are they related in this lecture?")

    if len(questions) < max_q:
        questions.append("Identify the main argument of the lecture and provide two supporting points.")

    return questions[:max_q]


def render_markdown(title: str,
                    keywords: list[str],
                    summary: list[str],
                    flashcards: list[tuple[str, str]],
                    questions: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"# {title}\n")

    lines.append("## Top Keywords / Phrases")
    for kw in keywords:
        lines.append(f"- {kw}")
    lines.append("")

    lines.append("## Summary")
    for s in summary:
        lines.append(f"- {s}")
    lines.append("")

    lines.append("## Flashcards")
    if not flashcards:
        lines.append("- *(No flashcards generated — try a longer lecture text.)*")
    else:
        for i, (front, back) in enumerate(flashcards, start=1):
            lines.append(f"**{i}. Q:** {front}")
            lines.append(f"- **A:** {back}\n")
    lines.append("")

    lines.append("## Exam-style Questions")
    for i, q in enumerate(questions, start=1):
        lines.append(f"{i}. {q}")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lecture Intelligence Toolkit: keywords, summary, flashcards, and exam-style questions (no external AI APIs)."
    )
    parser.add_argument("file", help="Path to lecture text/transcript file")
    parser.add_argument("--keywords", type=int, default=10, help="Number of keywords/phrases to extract (default: 10)")
    parser.add_argument("--summary", type=int, default=3, help="Number of summary sentences (default: 3)")
    parser.add_argument("--flashcards", type=int, default=10, help="Number of flashcards to generate (default: 10)")
    parser.add_argument("--questions", type=int, default=5, help="Number of exam-style questions (default: 5)")
    parser.add_argument("--out", type=str, default="", help="Optional output markdown file (e.g., output.md)")
    parser.add_argument("--title", type=str, default="Lecture Notes Output", help="Title for markdown output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: file not found: {file_path}")
        return 1

    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        print("Error: file is empty.")
        return 1

    ensure_nltk_resources()

    keywords = extract_keywords(text, top_k=args.keywords)
    summary = summarize_text(text, num_sentences=args.summary)
    flashcards = build_flashcards(text, keywords, max_cards=args.flashcards)
    questions = generate_exam_questions(keywords, summary, max_q=args.questions)

    print("=== Lecture Intelligence Toolkit ===")
    print("\nTop Keywords / Phrases:")
    for i, kw in enumerate(keywords, start=1):
        print(f"{i:>2}. {kw}")

    print("\nSummary:")
    for s in summary:
        print(f"- {s}")

    print("\nFlashcards:")
    if not flashcards:
        print("- (No flashcards generated — try a longer lecture text.)")
    else:
        for i, (front, back) in enumerate(flashcards, start=1):
            print(f"{i:>2}. Q: {front}")
            print(f"    A: {back}")

    print("\nExam-style Questions:")
    for i, q in enumerate(questions, start=1):
        print(f"{i:>2}. {q}")

    if args.out:
        md = render_markdown(args.title, keywords, summary, flashcards, questions)
        out_path = Path(args.out)
        out_path.write_text(md, encoding="utf-8")
        print(f"\nSaved markdown output to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

