from bs4 import BeautifulSoup
import pandas as pd

def scrape_questions(output_file):
    """Scrape questions data and save it to a CSV file."""
    try:
        # Load the HTML file
        with open("C:/Users/haadh/Downloads/Newest_Questions_Stack_Overflow.html", "r", encoding="utf-8") as file:

            soup = BeautifulSoup(file, "html.parser")

        # Extract questions
        questions = soup.select('.s-post-summary')
        questions_data = []
        for question in questions:
            title = question.select_one('.s-link').get_text(strip=True) if question.select_one('.s-link') else "N/A"
            votes = question.select_one('.s-post-summary--stats-item-number').get_text(strip=True) if question.select_one('.s-post-summary--stats-item-number') else "N/A"
            tags = [tag.get_text(strip=True) for tag in question.select('.post-tag')]
            questions_data.append({
                "Title": title,
                "Votes": votes,
                "Tags": ", ".join(tags)  # Join tags into a single string for CSV
            })

        # Save to CSV
        df = pd.DataFrame(questions_data)
        df.to_csv(output_file, index=False)
        print(f"Questions data successfully saved to {output_file}")

    except FileNotFoundError:
        print("Error: Questions page HTML file not found. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred while scraping questions: {e}")
