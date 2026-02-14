# Lecture Intelligence Toolkit (Study Pack Generator)

A lightweight NLP-based tool that converts lecture notes/transcripts into a structured study pack.
It generates:
- **Keywords / key phrases** (TF-IDF)
- **Extractive summary** (sentence scoring)
- **Flashcards** (keyword → best matching sentence)
- **Exam-style questions** (rule-based prompts)
- Optional **Markdown export** for easy copy/paste into notes

## Why this exists
Many lecture summarizers rely on paid LLMs. This project demonstrates a practical alternative using classic NLP techniques that are fast, transparent, and free to run locally.

## Features
- CLI tool with configurable output counts
- Works on any `.txt` transcript / lecture notes
- Markdown export for study workflows

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install scikit-learn nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

python lecture_ai.py sample_lecture.txt

