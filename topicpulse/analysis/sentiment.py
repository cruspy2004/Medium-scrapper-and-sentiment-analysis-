import google.generativeai as genai
import json
import os
from textblob import TextBlob

def classify_sentiment_batch(titles: list) -> list:
    if not titles:
        return []
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        return [_textblob_fallback(t) for t in titles]
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""
Classify the sentiment of each article title below.
Return a JSON array only — no markdown, no explanation.
Each item: {{"sentiment": "POSITIVE"|"NEGATIVE"|"NEUTRAL", "confidence": 0.0-1.0, "reason": "max 10 words"}}
Titles: {json.dumps(titles)}
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Remove markdown codeblock backticks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        return json.loads(text.strip())
    except Exception as e:
        print(f"Gemini Sentiment Error: {e}")
        return [_textblob_fallback(t) for t in titles]

def _textblob_fallback(title: str) -> dict:
    polarity = TextBlob(title).sentiment.polarity
    sentiment = "POSITIVE" if polarity > 0 else \
                "NEGATIVE" if polarity < 0 else "NEUTRAL"
    return {
        "sentiment": sentiment,
        "confidence": round(abs(polarity), 2),
        "reason": "TextBlob fallback used"
    }

def generate_topic_summary(topic: str, titles: list) -> str:
    if not titles:
        return "No articles found to summarize."
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        return "Configure a valid GEMINI_API_KEY to see AI-generated topic summaries."
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are analyzing content about "{topic}".
Here are recent article titles from Medium on this topic:
{json.dumps(titles[:30])}
In exactly 2-3 sentences, summarize what this community is currently
discussing, what problems they are solving, and whether the overall
tone is optimistic or concerned. Be specific and direct.
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Summary Error: {e}")
        return "Summary unavailable right now."
