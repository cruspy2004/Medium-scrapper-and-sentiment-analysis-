import pandas as pd

def calculate_opportunities(questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Opportunity score is calculated in the stackoverflow scraper:
    score = views / (answers + 1)
    Here we primarily just ensure it's sorted robustly.
    """
    if questions_df.empty:
        return pd.DataFrame()
        
    # Ensure opportunity_score exists
    if "opportunity_score" not in questions_df.columns:
        questions_df["opportunity_score"] = round(questions_df["views"] / (questions_df["answers"] + 1), 1)
        
    res = questions_df.sort_values(by="opportunity_score", ascending=False)
    return res
