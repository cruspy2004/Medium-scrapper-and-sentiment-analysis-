from bs4 import BeautifulSoup
import pandas as pd

def scrape_specific_question(output_file):
    """Scrape data from a specific question page and save it to a CSV file."""
    try:
        # Load the HTML file
        with open("C:/Users/haadh/Downloads/Is_it_possible_compile_time_check.html", "r", encoding="utf-8") as file:

            soup = BeautifulSoup(file, "html.parser")

        # Extract question details
        title = soup.select_one('.fs-headline1').get_text(strip=True) if soup.select_one('.fs-headline1') else "N/A"
        body = soup.select_one('.s-prose').get_text(strip=True) if soup.select_one('.s-prose') else "N/A"
        tags = [tag.get_text(strip=True) for tag in soup.select('.post-tag')]
        answers = [answer.get_text(strip=True) for answer in soup.select('.answer .s-prose')]

        # Prepare data
        question_data = [{
            "Title": title,
            "Body": body,
            "Tags": ", ".join(tags),
            "Answers": " | ".join(answers)  # Combine answers into a single string
        }]

        # Save to CSV
        df = pd.DataFrame(question_data)
        df.to_csv(output_file, index=False)
        print(f"Specific question data successfully saved to {output_file}")

    except FileNotFoundError:
        print("Error: Specific question page HTML file not found. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred while scraping the specific question: {e}")
