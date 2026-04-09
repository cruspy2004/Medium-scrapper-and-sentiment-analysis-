# TopicPulse v2.0 — Content Gap Finder 🚀

TopicPulse is a premium content intelligence tool that identifies underserved topics at the intersection of **Medium** engagement and **Stack Overflow** demand. 

Instead of just showing what's popular, TopicPulse shows you what's **missing**. It cross-references community sentiment with unanswered technical demand to surface the 5 best content opportunities for developers and technical founders.

## 🌟 Features

- **Unified Topic Scan:** Cross-reference Medium + Stack Overflow in one click.
- **Content Gap Scoring:** Weighted algorithm (60% SO Demand, 30% Sentiment, 10% Volume).
- **AI-Generated Briefs:** Get suggested titles, target audience, and key points via Gemini 2.0.
- **Sentiment Analytics:** Real-time classification of article titles (Positive/Negative/Neutral).
- **Premium UI:** Glassmorphism dark-mode dashboard with interactive charts.
- **CSV Export:** Download full datasets for offline analysis.

## 🛠️ Tech Stack

- **Backend:** Python / Flask
- **AI:** Google Gemini 1.5/2.0 Flash
- **Data:** Pandas (Scoring), BeautifulSoup (Scraping)
- **Frontend:** Vanilla CSS, Chart.js, Inter Typeface

---

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
- Python 3.9+
- A Google Gemini API Key (get it at [AI Studio](https://aistudio.google.com/))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/cruspy2004/Medium-scrapper-and-sentiment-analysis-.git
cd Medium-scrapper-and-sentiment-analysis-/topicpulse

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the `topicpulse/` directory:
```env
GEMINI_API_KEY=your_actual_key_here
SECRET_KEY=a_random_secure_string
FLASK_ENV=development
```

### 4. Run the App
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## 🌐 Deployment (Vercel)

Vercel supports Python/Flask natively. To deploy TopicPulse:

### 1. Configure for Vercel
Add a `vercel.json` to your project root (ensure it points to your app entry point):
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/app.py" }
  ]
}
```

### 2. Rename for Entry Point (Optional)
Vercel's Python runtime often looks for `index.py` or uses the file specified in the rewrite. Ensure `app` is the variable name for your Flask instance.

### 3. Deploy via CLI or Git
- **Git:** Push your code to GitHub and connect the repository in the Vercel Dashboard.
- **CLI:** Run `vercel` from the `topicpulse` directory.

### 4. Add Environment Variables
In the Vercel Dashboard under **Settings > Environment Variables**, add:
- `GEMINI_API_KEY`
- `SECRET_KEY`

---

## 📊 How the Gap Score Works
The "Content Gap Score" (0.0–1.0) is computed as follows:
- **SO Signal (60%):** Sum of `views / (answers + 1)` normalized.
- **Sentiment Penalty (30%):** Calculated based on negative/neutral sentiment (Negative = higher opportunity).
- **Volume Penalty (10%):** Inverse of existing article count.

**Difficulty Score:** Mapped 1-5 as the inverse of the Gap Score.

---

## 📜 License
Internal use for BSCS Class of 2027 project. 

*Author: Muhammad Haadhee Sheeraz Mian*
