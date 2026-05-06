import pandas as pd


def compute_gap_scores(articles: list, questions: list) -> list:
    """
    Cross-references Medium article saturation with Stack Overflow demand
    to produce a ranked list of content gaps.

    v3 gap score formula (quality-adjusted):
        gap_score = (quality_adjusted_so_score × 0.50)
                  + (sentiment_penalty × 0.30)
                  + (volume_penalty × 0.10)
                  + (answerability_signal × 0.10)

    Returns top 5 gaps sorted by gap_score descending.
    """
    if not articles and not questions:
        return []

    art_df = pd.DataFrame(articles) if articles else pd.DataFrame()
    q_df = pd.DataFrame(questions) if questions else pd.DataFrame()

    # --- Stack Overflow signal (quality-adjusted) ---
    so_signal = 0.5  # default if no questions
    answerability_signal = 0.0
    top_questions = []
    quality_breakdown = {"high": 0, "medium": 0, "low": 0}

    if not q_df.empty:
        if "opportunity_score" not in q_df.columns:
            q_df["opportunity_score"] = q_df["views"] / (q_df["answers"] + 1)
        max_opp = q_df["opportunity_score"].max() or 1

        # v3: Quality-adjusted scoring — only count well-formed unanswered questions
        if "quality_label" in q_df.columns:
            quality_mask = q_df["quality_label"].isin(["HIGH", "MEDIUM"])
            unanswered_mask = q_df["answers"] == 0
            genuine_gaps = q_df[quality_mask & unanswered_mask]

            so_signal = (
                genuine_gaps["opportunity_score"].sum() / (max_opp * max(len(genuine_gaps), 1))
                if len(genuine_gaps) > 0 else 0.0
            )
            so_signal = min(so_signal, 1.0)

            # Answerability signal: proportion of genuine gaps
            answerability_signal = (
                len(genuine_gaps) / len(q_df) if len(q_df) > 0 else 0.0
            )

            # Quality breakdown for all questions
            quality_breakdown = {
                "high": int((q_df["quality_label"] == "HIGH").sum()),
                "medium": int((q_df["quality_label"] == "MEDIUM").sum()),
                "low": int((q_df["quality_label"] == "LOW").sum()),
            }
        else:
            # Fallback for v2 compatibility (no quality_label)
            so_signal = min(q_df["opportunity_score"].sum() / (max_opp * len(q_df)), 1.0)

        # Keep top questions for brief context (prefer quality-adjusted sort)
        sort_col = "opportunity_score"
        top_questions = (
            q_df.sort_values(sort_col, ascending=False)
            .head(10)
            .to_dict("records")
        )

    # --- Medium signals — group articles by tag (subtopic) ---
    if art_df.empty:
        # No articles = everything is a gap, use SO data only
        gap_score = so_signal * 0.50 + 0.5 * 0.30 + 1.0 * 0.10 + answerability_signal * 0.10
        return [{
            "gap_title": questions[0].get("tag", "Unknown") if questions else "Unknown",
            "gap_score": round(gap_score, 3),
            "difficulty": _score_to_difficulty(gap_score),
            "so_signal": round(so_signal, 3),
            "answerability_signal": round(answerability_signal, 3),
            "sentiment_signal": "NEUTRAL",
            "article_count": 0,
            "quality_breakdown": quality_breakdown,
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

        # v3: Per-tag quality breakdown
        tag_quality = {"high": 0, "medium": 0, "low": 0}
        if not q_df.empty and "quality_label" in q_df.columns and "tags" in q_df.columns:
            tag_questions = q_df[q_df["tags"].apply(
                lambda t: tag.lower() in [x.lower() for x in t]
                if isinstance(t, list) else False
            )]
            tag_quality = {
                "high": int((tag_questions["quality_label"] == "HIGH").sum()),
                "medium": int((tag_questions["quality_label"] == "MEDIUM").sum()),
                "low": int((tag_questions["quality_label"] == "LOW").sum()),
            }

        gaps.append({
            "tag": tag,
            "so_signal": round(so_signal, 3),
            "answerability_signal": round(answerability_signal, 3),
            "sentiment_penalty": round(sentiment_penalty, 3),
            "volume_raw": total,
            "article_count": total,
            "sample_titles": group["title"].head(5).tolist(),
            "sentiment_signal": dominant,
            "quality_breakdown": tag_quality if any(tag_quality.values()) else quality_breakdown,
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

    # v3: Composite gap score (quality-adjusted weighted formula)
    gaps_df["gap_score"] = (
        gaps_df["so_signal"] * 0.50 +
        gaps_df["sentiment_penalty"] * 0.30 +
        gaps_df["volume_penalty"] * 0.10 +
        gaps_df["answerability_signal"] * 0.10
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

