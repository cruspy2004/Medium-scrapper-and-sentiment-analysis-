import pandas as pd

def rank_authors(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups articles by author and calculates influence score: 
    influence_score = (total_claps × 2) + total_responses
    """
    if articles_df.empty:
        return pd.DataFrame()
        
    author_df = articles_df.groupby('author').agg(
        total_claps=('claps', 'sum'),
        total_responses=('responses', 'sum'),
        article_count=('title', 'count')
    ).reset_index()

    author_df['influence_score'] = (author_df['total_claps'] * 2) + author_df['total_responses']
    author_df['profile_url'] = author_df['author'].apply(lambda x: f"https://medium.com/@{x.replace(' ', '').lower()}")
    
    author_df = author_df.sort_values('influence_score', ascending=False)
    return author_df
