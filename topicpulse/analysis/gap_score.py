import pandas as pd


def compute_gap_scores(articles: list, questions: list) -> list:
    """
    Cross-references Medium article saturation with Stack Overflow demand
    to produce a ranked list of content gaps.

    Gap score formula:
        gap_score = (so_opportunity_score × 0.6)
                  + (sentiment_penalty × 0.3)
                  + (volume_penalty × 0.1)

    Returns top 5 gaps sorted by gap_score descending.
    """
    if not articles and not questions:
        return []

    art_df = pd.DataFrame(articles) if articles else pd.DataFrame()
    q_df = pd.DataFrame(questions) if questions else pd.DataFrame()

    # --- Stack Overflow signal ---
    so_signal = 0.5  # default if no questions
    top_questions = []
    if not q_df.empty:
        if "opportunity_score" not in q_df.columns:
            q_df["opportunity_score"] = q_df["views"] / (q_df["answers"] + 1)
        max_opp = q_df["opportunity_score"].max() or 1
        so_signal = min(q_df["opportunity_score"].sum() / (max_opp * len(q_df)), 1.0)
        # Keep top questions for brief context
        top_questions = (
            q_df.sort_values("opportunity_score", ascending=False)
            .head(10)
            .to_dict("records")
        )

    # --- Medium signals — group articles by tag (subtopic) ---
    if art_df.empty:
        # No articles = everything is a gap, use SO data only
        return [{
            "gap_title": questions[0].get("tag", "Unknown") if questions else "Unknown",
            "gap_score": round(so_signal * 0.6 + 0.5 * 0.3 + 1.0 * 0.1, 3),
            "difficulty": _score_to_difficulty(so_signal * 0.6 + 0.5 * 0.3 + 1.0 * 0.1),
            "so_signal": round(so_signal, 3),
            "sentiment_signal": "NEUTRAL",
            "article_count": 0,
            "top_questions": top_questions[:5],
            "sample_titles": [],
            "n_pos": 0, "n_neg": 0, "n_neu": 0,
        }]

    grouped = art_df.groupby("tag")
    gaps = []

    for tag, group in grouped:
        sentiment_counts = group["sentiment"].value_counts()
        n_pos = int(sentiment_counts.get("POSITIVE", 0))
        n_neg = int(sentiment_counts.get("NEGATIVE", 0))
        n_neu = int(sentiment_counts.get("NEUTRAL", 0))
        total = len(group)

        # Sentiment penalty: high negative = bigger gap = bigger opportunity
        sentiment_penalty = (n_neg * 1.0 + n_neu * 0.5) / total if total else 0.5

        # Dominant sentiment for display
        dominant = max(
            [("POSITIVE", n_pos), ("NEGATIVE", n_neg), ("NEUTRAL", n_neu)],
            key=lambda x: x[1]
        )[0]

        gaps.append({
            "tag": tag,
            "so_signal": round(so_signal, 3),
            "sentiment_penalty": round(sentiment_penalty, 3),
            "volume_raw": total,
            "article_count": total,
            "sample_titles": group["title"].head(5).tolist(),
            "sentiment_signal": dominant,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_neu": n_neu,
            "top_questions": top_questions[:5],
        })

    if not gaps:
        return []

    gaps_df = pd.DataFrame(gaps)

    # Normalize volume penalty: fewer articles = bigger gap
    max_vol = gaps_df["volume_raw"].max() or 1
    gaps_df["volume_penalty"] = 1 - (gaps_df["volume_raw"] / max_vol)

    # Composite gap score (weighted)
    gaps_df["gap_score"] = (
        gaps_df["so_signal"] * 0.6 +
        gaps_df["sentiment_penalty"] * 0.3 +
        gaps_df["volume_penalty"] * 0.1
    ).round(3)

    # Difficulty: inverse of gap score, mapped to 1-5
    gaps_df["difficulty"] = gaps_df["gap_score"].apply(_score_to_difficulty)

    # Use tag as gap_title
    gaps_df["gap_title"] = gaps_df["tag"]

    # Select and sort
    result = (
        gaps_df.sort_values("gap_score", ascending=False)
        .head(5)
        .to_dict("records")
    )

    # Clean up internal columns not needed by frontend
    for gap in result:
        gap.pop("tag", None)
        gap.pop("volume_raw", None)
        gap.pop("sentiment_penalty", None)
        gap.pop("volume_penalty", None)

    return result


def _score_to_difficulty(score: float) -> int:
    """Map gap score (0-1) to difficulty (1-5). High gap = low difficulty."""
    d = round(5 - (score * 4))
    return max(1, min(5, d))
