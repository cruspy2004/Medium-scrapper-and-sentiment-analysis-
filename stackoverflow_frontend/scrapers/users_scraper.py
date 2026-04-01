from bs4 import BeautifulSoup
import pandas as pd

def scrape_users(output_file):
    with open("C:/Users/haadh/Downloads/Users_Stack_Overflow.html", "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    users = soup.select(".user-details")
    users_data = []
    for user in users:
        username = user.select_one("a").get_text(strip=True) if user.select_one("a") else "N/A"
        reputation = user.find_next_sibling("div").get_text(strip=True) if user.find_next_sibling("div") else "N/A"
        users_data.append({"Username": username, "Reputation": reputation})

    df = pd.DataFrame(users_data)
    df.to_csv(output_file, index=False)
