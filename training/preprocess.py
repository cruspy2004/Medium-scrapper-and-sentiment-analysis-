"""
TopicPulse v3 — Preprocessing Pipeline
Handles data loading, HTML stripping, tokenization, vocabulary building,
label engineering, and train/val/test splitting for Stack Overflow questions.
"""

import os
import re
import json
import html
import logging
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure NLTK data is available
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    ("tokenizers", "punkt_tab"),
    ("corpora", "stopwords"),
    ("corpora", "wordnet"),
]
for _category, _resource in _NLTK_RESOURCES:
    try:
        nltk.data.find(f"{_category}/{_resource}")
    except Exception:
        try:
            nltk.download(_resource, quiet=True)
        except Exception:
            pass  # Best-effort; tokenize_text will use fallback if needed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEXT_VOCAB_SIZE = 30000
CODE_VOCAB_SIZE = 10000
TEXT_MAX_LEN = 300
CODE_MAX_LEN = 200

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Common programming keywords to preserve during code tokenization
CODE_KEYWORDS = {
    "def", "class", "import", "from", "return", "if", "else", "elif", "for",
    "while", "try", "except", "with", "as", "yield", "lambda", "raise",
    "pass", "break", "continue", "True", "False", "None", "and", "or", "not",
    "in", "is", "self", "print", "len", "range", "list", "dict", "set",
    "int", "str", "float", "bool", "async", "await", "function", "var",
    "let", "const", "this", "new", "typeof", "instanceof", "null",
    "undefined", "void", "public", "private", "static", "final",
    "abstract", "interface", "extends", "implements", "throws",
    "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "JOIN",
    "GROUP", "ORDER", "BY", "HAVING", "CREATE", "TABLE", "INDEX",
}


# ===================================================================
# DATA LOADING
# ===================================================================

def load_stackoverflow_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a Stack Exchange Data Explorer CSV dump.

    Expected columns: Id, Title, Body, Tags, CreationDate, Score,
    ViewCount, AnswerCount, AcceptedAnswerId, (optional) FirstAnswerDate
    """
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize column names (SEDE exports can vary in casing)
    df.columns = [c.strip() for c in df.columns]

    required = {"Title", "Body", "Tags", "CreationDate", "AnswerCount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    logger.info("Loaded %d questions", len(df))
    return df


def generate_synthetic_dataset(n_samples: int = 5000,
                                output_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic dataset for development/testing when real
    Stack Exchange data is unavailable.
    """
    logger.info("Generating synthetic dataset with %d samples", n_samples)
    rng = np.random.RandomState(42)

    titles = [
        "How to implement binary search in Python",
        "React useState not updating state",
        "Docker container keeps crashing on startup",
        "SQL JOIN performance optimization",
        "Machine learning model overfitting",
        "Kubernetes pod scheduling issues",
        "JavaScript async await error handling",
        "Flask REST API authentication",
        "TensorFlow GPU memory allocation",
        "Git merge conflict resolution",
    ]

    code_snippets = [
        "<code>def binary_search(arr, target):\n    left, right = 0, len(arr)-1\n    while left <= right:\n        mid = (left+right)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid+1\n        else: right = mid-1\n    return -1</code>",
        "<code>const [state, setState] = useState(0);\nuseEffect(() => { setState(prev => prev + 1); }, []);</code>",
        "<pre>FROM python:3.9\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"app.py\"]</pre>",
        "<code>SELECT a.*, b.name FROM orders a\nINNER JOIN customers b ON a.customer_id = b.id\nWHERE a.total > 100\nGROUP BY b.name;</code>",
        "",
    ]

    bodies = [
        "<p>I'm trying to implement a binary search algorithm but getting wrong results. "
        "Here is my code:</p>" + code_snippets[0],
        "<p>My React component state isn't updating as expected. "
        "When I call setState, the value doesn't change immediately.</p>" + code_snippets[1],
        "<p>I've created a Dockerfile but the container crashes immediately. "
        "Here's my configuration:</p>" + code_snippets[2],
        "<p>My SQL query is running very slowly on large datasets. "
        "How can I optimize this JOIN operation?</p>" + code_snippets[3],
        "<p>My neural network is severely overfitting the training data. "
        "What techniques can I use to reduce overfitting?</p>",
    ]

    tag_options = [
        "<python><algorithm><binary-search>",
        "<javascript><reactjs><hooks>",
        "<docker><containers><devops>",
        "<sql><mysql><performance>",
        "<machine-learning><deep-learning><tensorflow>",
        "<kubernetes><containers>",
        "<javascript><async-await>",
        "<python><flask><rest-api>",
        "<tensorflow><gpu><memory>",
        "<git><version-control>",
    ]

    records = []
    for i in range(n_samples):
        idx = i % len(titles)
        has_answer = rng.random() > 0.35
        answer_count = rng.randint(1, 15) if has_answer else 0

        creation = datetime(
            2023, rng.randint(1, 13) % 12 + 1, rng.randint(1, 29) % 28 + 1,
            rng.randint(0, 24) % 24, rng.randint(0, 60) % 60
        )

        hours_to_answer = rng.choice([0.5, 2, 12, 48, 168, 500, -1],
                                      p=[0.15, 0.25, 0.2, 0.15, 0.1, 0.1, 0.05])

        records.append({
            "Id": i + 1,
            "Title": titles[idx] + f" (variant {i})",
            "Body": bodies[idx % len(bodies)],
            "Tags": tag_options[idx % len(tag_options)],
            "CreationDate": creation.isoformat(),
            "Score": rng.randint(-5, 100),
            "ViewCount": rng.randint(10, 100000),
            "AnswerCount": answer_count,
            "AcceptedAnswerId": rng.randint(1000, 99999) if has_answer and rng.random() > 0.3 else None,
            "FirstAnswerDate": (
                (creation + pd.Timedelta(hours=float(hours_to_answer))).isoformat()
                if has_answer and hours_to_answer > 0 else None
            ),
        })

    df = pd.DataFrame(records)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Synthetic data saved to %s", output_path)
    return df


# ===================================================================
# HTML / CODE EXTRACTION
# ===================================================================

def extract_code_blocks(html_body: str) -> list:
    """Extract <code> and <pre> tag contents from HTML body."""
    if not html_body or not isinstance(html_body, str):
        return []
    soup = BeautifulSoup(html_body, "html.parser")
    blocks = []
    for tag in soup.find_all(["code", "pre"]):
        text = tag.get_text(strip=True)
        if text:
            blocks.append(text)
    return blocks


def strip_html(html_body: str) -> str:
    """Remove HTML tags from body, returning plain text."""
    if not html_body or not isinstance(html_body, str):
        return ""
    soup = BeautifulSoup(html_body, "html.parser")
    # Remove code blocks first
    for tag in soup.find_all(["code", "pre"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = html.unescape(text)
    # Clean up URLs and excessive whitespace
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_tags(tags_str: str) -> list:
    """Parse SO tag format '<python><flask>' into ['python', 'flask']."""
    if not tags_str or not isinstance(tags_str, str):
        return []
    return re.findall(r"<([^>]+)>", tags_str)


# ===================================================================
# TOKENIZATION
# ===================================================================

def tokenize_text(text: str) -> list:
    """
    Tokenize text: lowercase, word_tokenize, remove stopwords,
    lemmatize.
    """
    if not text:
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in STOP_WORDS and len(t) > 1
    ]
    return tokens


def tokenize_code(code_text: str) -> list:
    """
    Tokenize code preserving language keywords, operators, and
    identifiers.
    """
    if not code_text:
        return []
    # Split on whitespace and common delimiters but keep operators
    tokens = re.findall(r"[a-zA-Z_]\w*|[+\-*/=<>!&|^~%]+|[{}()\[\];,.:@#]|\d+", code_text)
    result = []
    for t in tokens:
        if t in CODE_KEYWORDS:
            result.append(t)
        elif t.isidentifier():
            # Split camelCase and snake_case
            parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", t).split()
            for p in parts:
                for sub in p.split("_"):
                    if sub:
                        result.append(sub.lower())
        elif re.match(r"^[+\-*/=<>!&|^~%]+$", t):
            result.append(t)
        elif t.isdigit():
            result.append("<NUM>")
    return result


# ===================================================================
# VOCABULARY BUILDING
# ===================================================================

def build_vocabulary(token_lists: list, max_size: int) -> dict:
    """
    Build word-to-index vocabulary from a list of token lists.
    Index 0 = <PAD>, 1 = <UNK>.
    """
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)

    return vocab


def tokens_to_indices(tokens: list, vocab: dict, max_len: int) -> list:
    """Convert token list to padded/truncated index list."""
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    # Pad to max_len
    indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices


# ===================================================================
# LABEL ENGINEERING
# ===================================================================

def engineer_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary answerability label and time bucket label.
    """
    # Binary: has accepted answer
    df["has_accepted_answer"] = (
        df["AcceptedAnswerId"].notna().astype(int)
    )

    # Time-to-first-answer buckets
    df["time_bucket"] = _compute_time_buckets(df)

    return df


def _compute_time_buckets(df: pd.DataFrame) -> pd.Series:
    """Compute time-to-first-answer buckets."""
    buckets = []
    for _, row in df.iterrows():
        if pd.isna(row.get("FirstAnswerDate")) or row.get("AnswerCount", 0) == 0:
            buckets.append(3)  # over_7d_or_never
        else:
            try:
                created = pd.to_datetime(row["CreationDate"])
                answered = pd.to_datetime(row["FirstAnswerDate"])
                hours = (answered - created).total_seconds() / 3600

                if hours < 1:
                    buckets.append(0)   # under_1h
                elif hours < 24:
                    buckets.append(1)   # 1h_to_24h
                elif hours < 168:       # 7 days
                    buckets.append(2)   # 1d_to_7d
                else:
                    buckets.append(3)   # over_7d_or_never
            except Exception:
                buckets.append(3)

    return pd.Series(buckets, index=df.index)


# ===================================================================
# METADATA FEATURES
# ===================================================================

def extract_metadata(row: pd.Series) -> list:
    """
    Extract 8 metadata features for a single question.
    Returns: [tag_count, title_length, body_length, code_length,
              has_code, num_code_blocks, hour_of_day, is_weekend]
    """
    tags = parse_tags(row.get("Tags", ""))
    body_text = strip_html(row.get("Body", ""))
    code_blocks = extract_code_blocks(row.get("Body", ""))

    code_text = " ".join(code_blocks)
    title = row.get("Title", "")

    try:
        dt = pd.to_datetime(row.get("CreationDate"))
        hour = dt.hour
        is_weekend = 1 if dt.weekday() >= 5 else 0
    except Exception:
        hour = 12
        is_weekend = 0

    return [
        len(tags),                              # tag_count
        len(title.split()),                     # title_length
        len(body_text.split()),                 # body_length
        len(code_text.split()),                 # code_length
        1 if code_blocks else 0,                # has_code
        len(code_blocks),                       # num_code_blocks
        hour,                                   # hour_of_day
        is_weekend,                             # is_weekend
    ]


# ===================================================================
# FULL PREPROCESSING PIPELINE
# ===================================================================

def preprocess_dataset(df: pd.DataFrame, output_dir: str = "models"):
    """
    Full preprocessing pipeline.

    1. Extract code blocks and strip HTML
    2. Tokenize text and code
    3. Build vocabularies
    4. Engineer labels
    5. Extract metadata features
    6. Convert to padded integer sequences
    7. Split into train/val/test

    Returns:
        dict with keys: train, val, test — each a dict of numpy arrays
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Step 1: Extracting code blocks and stripping HTML...")
    df["code_blocks"] = df["Body"].apply(extract_code_blocks)
    df["body_text"] = df["Body"].apply(strip_html)
    df["title_text"] = df["Title"].fillna("")

    logger.info("Step 2: Tokenizing text and code...")
    df["text_tokens"] = (df["title_text"] + " " + df["body_text"]).apply(tokenize_text)
    df["code_tokens"] = df["code_blocks"].apply(
        lambda blocks: tokenize_code(" ".join(blocks))
    )

    logger.info("Step 3: Building vocabularies...")
    text_vocab = build_vocabulary(df["text_tokens"].tolist(), TEXT_VOCAB_SIZE)
    code_vocab = build_vocabulary(df["code_tokens"].tolist(), CODE_VOCAB_SIZE)

    # Save vocabularies
    text_vocab_path = os.path.join(output_dir, "vocab_text.json")
    code_vocab_path = os.path.join(output_dir, "vocab_code.json")
    with open(text_vocab_path, "w") as f:
        json.dump(text_vocab, f)
    with open(code_vocab_path, "w") as f:
        json.dump(code_vocab, f)
    logger.info("Vocabularies saved — text: %d tokens, code: %d tokens",
                len(text_vocab), len(code_vocab))

    logger.info("Step 4: Engineering labels...")
    df = engineer_labels(df)

    logger.info("Step 5: Extracting metadata features...")
    metadata_list = df.apply(extract_metadata, axis=1).tolist()

    logger.info("Step 6: Converting to padded sequences...")
    text_indices = np.array([
        tokens_to_indices(toks, text_vocab, TEXT_MAX_LEN)
        for toks in df["text_tokens"]
    ], dtype=np.int64)

    code_indices = np.array([
        tokens_to_indices(toks, code_vocab, CODE_MAX_LEN)
        for toks in df["code_tokens"]
    ], dtype=np.int64)

    metadata = np.array(metadata_list, dtype=np.float32)
    labels_answer = df["has_accepted_answer"].values.astype(np.float32)
    labels_time = df["time_bucket"].values.astype(np.int64)

    logger.info("Step 7: Splitting train/val/test (70/15/15)...")
    # Stratified by answerability label
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=42,
        stratify=labels_answer
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42,
        stratify=labels_answer[temp_idx]
    )

    def _subset(idx):
        return {
            "text": text_indices[idx],
            "code": code_indices[idx],
            "metadata": metadata[idx],
            "label_answer": labels_answer[idx],
            "label_time": labels_time[idx],
        }

    splits = {
        "train": _subset(train_idx),
        "val":   _subset(val_idx),
        "test":  _subset(test_idx),
    }

    logger.info("Split sizes — train: %d, val: %d, test: %d",
                len(train_idx), len(val_idx), len(test_idx))

    return splits, text_vocab, code_vocab


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # If no real CSV exists, generate synthetic data
    csv_path = os.environ.get("SO_DATA_CSV", "")
    if csv_path and os.path.exists(csv_path):
        df = load_stackoverflow_csv(csv_path)
    else:
        logger.info("No CSV found — generating synthetic dataset for testing.")
        df = generate_synthetic_dataset(5000, "data/synthetic_so.csv")

    splits, text_vocab, code_vocab = preprocess_dataset(df, output_dir="models")
    logger.info("Preprocessing complete!")
    logger.info("Text vocab size: %d", len(text_vocab))
    logger.info("Code vocab size: %d", len(code_vocab))
