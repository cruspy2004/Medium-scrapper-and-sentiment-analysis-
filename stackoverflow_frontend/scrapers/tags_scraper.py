from bs4 import BeautifulSoup
import pandas as pd

def scrape_tags(output_file):
    """Scrape tags data and save it to a CSV file."""
    try:
        # Load the HTML file
        with open("C:/Users/haadh/Downloads/Tags_Stack_Overflow.html", "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        # Extract tags
        tags = soup.select('.post-tag')
        tags_data = [{"Tag": tag.get_text(strip=True)} for tag in tags]

        # Save to CSV
        df = pd.DataFrame(tags_data)
        df.to_csv(output_file, index=False)
        print(f"Tags data successfully saved to {output_file}")

    except FileNotFoundError:
        print("Error: Tags page HTML file not found. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred while scraping tags: {e}")
