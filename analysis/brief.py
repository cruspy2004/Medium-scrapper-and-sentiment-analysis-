import google.generativeai as genai
import json
import os


def generate_brief(gap: dict) -> dict:
    """
    Generate a full content brief for a content gap using Gemini.
    Called on-demand when user clicks a gap card (not pre-cached).

    Returns JSON with: suggested_title, why_now, target_audience,
    key_points, questions_to_address, difficulty, difficulty_label.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        return {
            "error": "GEMINI_API_KEY not configured",
            "suggested_title": "Configure a valid GEMINI_API_KEY to generate briefs.",
            "why_now": "API key is missing.",
            "target_audience": "N/A",
            "key_points": ["Set up your Gemini API key in the .env file"],
            "questions_to_address": [],
            "difficulty": 3,
            "difficulty_label": "Unknown — API key not set"
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Build context from gap data
        topic = gap.get("topic", "Unknown")
        gap_title = gap.get("gap_title", "Unknown")
        gap_score = gap.get("gap_score", 0)
        n_pos = gap.get("n_pos", 0)
        n_neg = gap.get("n_neg", 0)
        n_neu = gap.get("n_neu", 0)
        sample_titles = gap.get("sample_titles", [])
        top_questions = gap.get("top_questions", [])

        prompt = f"""
You are a content strategist. Generate a content brief for the topic gap below.
Return JSON only — no markdown, no preamble, no code fences.

Gap context:
- Topic: {topic}
- Subtopic: {gap_title}
- Gap score: {gap_score} (higher = bigger opportunity)
- Medium sentiment: {n_pos} positive, {n_neg} negative, {n_neu} neutral articles
- Sample Medium titles: {json.dumps(sample_titles[:5])}
- Top Stack Overflow questions: {json.dumps(top_questions[:5])}

Return this exact JSON shape:
{{
  "suggested_title": "compelling, specific article title",
  "why_now": "2 sentences on why this gap exists and why now is the time to write about it",
  "target_audience": "specific description of who will read this",
  "key_points": ["point 1", "point 2", "point 3"],
  "questions_to_address": [
    {{"title": "SO question title", "views": 42000, "answers": 1, "url": "https://stackoverflow.com/..."}}
  ],
  "difficulty": {gap.get("difficulty", 3)},
  "difficulty_label": "short label explaining the difficulty score"
}}
"""
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Strip markdown code fences if Gemini wraps the response
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())

    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse Gemini response: {e}",
            "suggested_title": "Brief generation failed — Gemini returned invalid JSON. Try again.",
            "why_now": "The AI response could not be parsed.",
            "target_audience": "N/A",
            "key_points": ["Try clicking Generate Brief again"],
            "questions_to_address": gap.get("top_questions", [])[:3],
            "difficulty": gap.get("difficulty", 3),
            "difficulty_label": "Unknown"
        }
    except Exception as e:
        return {
            "error": str(e),
            "suggested_title": "Brief generation failed — try again.",
            "why_now": str(e),
            "target_audience": "N/A",
            "key_points": ["An error occurred during brief generation"],
            "questions_to_address": gap.get("top_questions", [])[:3],
            "difficulty": gap.get("difficulty", 3),
            "difficulty_label": "Unknown"
        }
