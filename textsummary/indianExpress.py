import json
import os
from selenium import webdriver
import mysql.connector

import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options



def scrape_indian_express(driver):
    driver.get("https://indianexpress.com/")
    time.sleep(5)
    list = []
    news_list = []

    try:
        # Find the right-part news section
        right_part = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, "right-part")))
        top_news = right_part.find_element(By.CLASS_NAME, "top-news")
        top_news_ul = top_news.find_element(By.TAG_NAME, "ul")
        top_news_ul_li = top_news_ul.find_elements(By.TAG_NAME, "li")

        for li in top_news_ul_li:
            try:
                headline = li.find_element(
                    By.TAG_NAME, "h3").find_element(By.TAG_NAME, "a")
                title = headline.text.strip()
                link = headline.get_attribute("href")

                print(title)
                print(link)

                # Open the link in the browser
                if title and link:
                    try:
                        list.append({"title": title, "link": link})
                    except Exception as e:
                        print(f"inside  if title and link: {e}")
            except Exception as e:
                print(f"Error processing a news item: {e}")

        news_list = scrape_inside_links(driver, list)

    except Exception as e:
        print(f"Error during scrape_indian_express: {e}")

    return news_list


def scrape_inside_links(driver, n_list):
    new_list = []

    for i in n_list:
        try:

            driver.get(i["link"])
            time.sleep(5)
            # Optional: log the link being opened
            print(f"Opening link: {i["link"]}")

            driver.get(i["link"])  # Open the news link
            time.sleep(5)

            heading_inside = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1")))
            print(heading_inside.text)

            mini_heading_inside = driver.find_elements(By.TAG_NAME, "h2")[0]
            print(mini_heading_inside.text)

            content = driver.find_elements(By.ID, "pcl-full-content")[0]
            # print(content.text)
            final_content = content.text
            # print(final_content)

            new_list.append(
                {"title": i["title"], "link": i["link"], "content": final_content})

        except Exception as e:
            print(f"inside try catch scrape_inside_links , {e}")

    return new_list


def scrape_latest_news_selenium(driver):
    driver.get("https://indianexpress.com/")
    time.sleep(5)
    news_list = []
    list = []

    try:
        # Find the "Latest News" section
        latest_news = driver.find_elements(By.CLASS_NAME, "right-part")[1]
        top_news = latest_news.find_element(By.CLASS_NAME, "top-news")
        top_news_ul = top_news.find_element(By.TAG_NAME, "ul")
        top_news_ul_li = top_news_ul.find_elements(By.TAG_NAME, 'li')

        for li in top_news_ul_li:
            try:
                headline = li.find_element(
                    By.TAG_NAME, "h3").find_element(By.TAG_NAME, "a")
                title = headline.text.strip()
                link = headline.get_attribute("href")

                # Add the news to the list
                if title and link:
                    list.append({"title": title, "link": link})

            except Exception as e:
                print(f"Error processing a news item: {e}")

        news_list = scrape_inside_links(driver, list)

    except Exception as e:
        print(f"Error during scrape_latest_news_selenium: {e}")

    return news_list


def scrape_top_news_selenium(driver):
    driver.get("https://indianexpress.com/")
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "left-part")))
    news_list = []
    list = []

    try:
        left_part = driver.find_elements(By.CLASS_NAME, "left-part")
        if len(left_part) < 2:
            print("Left part not found or does not have enough sections.")
            return news_list

        left_part = left_part[1]
        left_part_divs = left_part.find_elements(By.TAG_NAME, "div")

        print(f"Found {len(left_part_divs)} articles.")

        for article in left_part_divs:
            try:
                second_div_inside_div = article.find_elements(
                    By.TAG_NAME, "div")
                if len(second_div_inside_div) < 2:
                    print("Not enough inner divs found in this article.")
                    continue

                # Check if the second inner div contains h3 tags
                second_div_inside_div_h3 = second_div_inside_div[1].find_elements(
                    By.TAG_NAME, "h3")
                if not second_div_inside_div_h3:
                    print("No h3 tags found in this inner div.")
                    continue

                # Check if there are anchor tags within h3
                link_element = second_div_inside_div_h3[0].find_elements(
                    By.TAG_NAME, "a")
                if not link_element:
                    print("No anchor tags found in h3.")
                    continue

                link = link_element[0].get_attribute("href")
                title = link_element[0].text

                print(f"Title: {title}")
                print(f"Link: {link}")

                if title and link:
                    list.append({"title": title, "link": link})

            except Exception as e:
                print(f"Error processing article: {e}")

        news_list = scrape_inside_links(driver, list)

    except Exception as e:
        print(f"Error during main scraping process: {e}")

    return news_list


def save_news_to_file(driver, file_path="indian_express_all_news.json"):
    # Step 1: Check if the file exists, and load the existing news if available
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            all_news = json.load(file)
    else:
        all_news = []

    # Step 2: Scrape the news from all sections using the same driver instance
    all_news.extend(scrape_indian_express(driver))
    all_news.extend(scrape_latest_news_selenium(driver))
    all_news.extend(scrape_top_news_selenium(driver))

    # Step 3: Remove duplicates based on link
    seen_links = set()
    unique_news = []
    for news in all_news:
        if news['link'] not in seen_links:
            unique_news.append(news)
            seen_links.add(news['link'])

    # Step 4: Save the combined news to the file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(unique_news, file, ensure_ascii=False, indent=4)
    store_news_in_db(unique_news)
    print(f"Scraped and saved {len(unique_news)} articles to '{file_path}'.")


def store_news_in_db(news_list):
    # Connect to MySQL database
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345678",
        database="news_db"
    )

    cursor = mydb.cursor()

    for news_item in news_list:
        title = news_item['title']
        link = news_item['link']
        content = news_item['content']

        # Insert into the database
        try:
            cursor.execute("INSERT INTO news (title, link, content) VALUES (%s, %s, %s)",
                           (title, link, content))
            mydb.commit()  # Commit the transaction
            print(f"Inserted: {title}")
        except mysql.connector.Error as err:
            print(f"Error inserting data: {err}")

    cursor.close()
    mydb.close()

    return news_list


def main():
    
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Firefox(options=options)
    # driver = webdriver.Firefox()

    try:
        save_news_to_file(driver)
    finally:
        driver.quit()


# Run the main function
if __name__ == "__main__":
    main()
