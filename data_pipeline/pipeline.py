import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
import sqlite3

BASE_URL = "https://books.toscrape.com/"
RATING_WORDS = {"One", "Two", "Three", "Four", "Five"}

session = requests.Session()
session.headers.update({"User-Agent": "ZeptoCapstoneStudentProject/1.0"})


def get_soup(url):
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


books = []

for page_number in range(1, 6):
    if page_number == 1:
        listing_url = BASE_URL
    else:
        listing_url = urljoin(BASE_URL, f"catalogue/page-{page_number}.html")

    print(f"Reading catalogue page {page_number}...")
    listing_soup = get_soup(listing_url)

    for book in listing_soup.select("article.product_pod"):
        title_link = book.select_one("h3 a")
        price_tag = book.select_one(".price_color")
        availability_tag = book.select_one(".availability")
        rating_tag = book.select_one(".star-rating")

        if not all([title_link, price_tag, availability_tag, rating_tag]):
            continue

        title = title_link.get("title", title_link.get_text(strip=True))
        book_url = urljoin(listing_url, title_link.get("href", ""))

        rating = next(
            (word for word in rating_tag.get("class", []) if word in RATING_WORDS),
            "Unknown",
        )

        # The category is listed on each book's detail page.
        try:
            detail_soup = get_soup(book_url)
            category_tag = detail_soup.select_one(
                "ul.breadcrumb li:nth-of-type(3) a"
            )
            category = category_tag.get_text(strip=True) if category_tag else "Unknown"
        except requests.RequestException:
            category = "Unknown"

        books.append(
            {
                "title": title,
                "price": price_tag.get_text(strip=True),
                "star_rating": rating,
                "availability": availability_tag.get_text(" ", strip=True),
                "category": category,
            }
        )

        time.sleep(0.1)  # brief pause between requests

df = pd.DataFrame(books)
df.to_csv("data_pipeline/raw_books.csv", index=False)

print(f"Saved {len(df)} books to data_pipeline/raw_books.csv")
print(df.head())
# Convert scraped text into the required data types.
df["price_gbp"] = pd.to_numeric(
    df["price"].str.replace(r"[^0-9.]", "", regex=True),
    errors="coerce",
)

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["rating"] = df["star_rating"].map(rating_map)

# Use the median when a numeric value could not be parsed.
df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
df["rating"] = df["rating"].fillna(df["rating"].median()).round().astype(int)

df["in_stock"] = df["availability"].str.contains(
    "in stock", case=False, na=False
)

# Required project conversion rate: 1 GBP = 105.50 INR.
df["price_inr"] = (df["price_gbp"] * 105.50).round(2)

# Keep the useful raw fields alongside the cleaned columns.
df.to_csv("data_pipeline/cleaned_books.csv", index=False)

print("\nCleaned data:")
print(df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]].head())
print(f"\nRows: {len(df)} | Categories: {df['category'].nunique()}")
df["price_gbp"] = pd.to_numeric(
    df["price"].str.replace(r"[^0-9.]", "", regex=True),
    errors="coerce",
)
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["rating"] = df["star_rating"].map(rating_map)

df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
df["rating"] = df["rating"].fillna(df["rating"].median()).round().astype(int)
df["in_stock"] = df["availability"].str.contains("in stock", case=False, na=False)
df["price_inr"] = (df["price_gbp"] * 105.50).round(2)

df.to_csv("data_pipeline/cleaned_books.csv", index=False)
print("\nCleaned data:")
print(df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]].head())
print(f"\nRows: {len(df)} | Categories: {df['category'].nunique()}")
# Create related category and book tables, then load the cleaned data.
categories_df = pd.DataFrame(
    {"category_name": sorted(df["category"].dropna().unique())}
)
categories_df["category_id"] = range(1, len(categories_df) + 1)

category_ids = dict(
    zip(categories_df["category_name"], categories_df["category_id"])
)
books_df = df.copy()
books_df["category_id"] = books_df["category"].map(category_ids)
books_df["book_id"] = range(1, len(books_df) + 1)

connection = sqlite3.connect("data_pipeline/zepto_books.db")
connection.execute("PRAGMA foreign_keys = ON")

connection.executescript("""
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
""")

connection.executemany(
    "INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
    list(zip(categories_df["category_id"], categories_df["category_name"])),
)

connection.executemany(
    """
    INSERT INTO books
        (book_id, title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    [
        (
            row.book_id,
            row.title,
            row.price_gbp,
            row.price_inr,
            row.rating,
            int(row.in_stock),
            row.category_id,
        )
        for row in books_df.itertuples()
    ],
)

connection.commit()

book_count = pd.read_sql_query("SELECT COUNT(*) AS book_count FROM books", connection)
category_count = pd.read_sql_query(
    "SELECT COUNT(*) AS category_count FROM categories", connection
)

print("\nSQLite database created: data_pipeline/zepto_books.db")
print(book_count)
print(category_count)

connection.close()
connection = sqlite3.connect("data_pipeline/zepto_books.db")
# Run five SQL queries that demonstrate the required clauses.
queries = {
    "1. Books over GBP 30, ordered by price, limited to 10": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp > 30
        ORDER BY price_gbp DESC
        LIMIT 10
    """,
    "2. Distinct categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
    """,
    "3. Books rated 4 or 5 (IN)": """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 10
    """,
    "4. Books priced between GBP 20 and GBP 40 (BETWEEN)": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
        LIMIT 10
    """,
}

join_query = """
    SELECT c.category_name, b.title, b.rating, b.price_gbp
    FROM books AS b
    JOIN categories AS c ON b.category_id = c.category_id
    ORDER BY b.rating DESC, b.price_gbp ASC, b.title ASC
    LIMIT 10
"""

query_results = {}

for query_name, query_text in queries.items():
    result = pd.read_sql_query(query_text, connection)
    query_results[query_name] = result
    print(f"\n{query_name}\n{result.to_string(index=False)}")

# Read the JOIN result from SQLite.
sql_join_result = pd.read_sql_query(join_query, connection)
print(f"\n5. JOIN query — SQLite result\n{sql_join_result.to_string(index=False)}")

# Recreate the same JOIN from the in-memory pandas DataFrames.
merge_join_result = pd.merge(books_df, categories_df, on="category_id")
merge_join_result = merge_join_result[
    ["category_name", "title", "rating", "price_gbp"]
].sort_values(
    ["rating", "price_gbp", "title"],
    ascending=[False, True, True],
).head(10).reset_index(drop=True)

print(f"\nJOIN query — pandas merge result\n{merge_join_result.to_string(index=False)}")
print("\nDo the SQL JOIN and pandas merge match?",
      sql_join_result.equals(merge_join_result))

# Save every query and its result as a Markdown text file.
with open("data_pipeline/sql_query_outputs.md", "w", encoding="utf-8") as output_file:
    for query_name, query_text in queries.items():
        output_file.write(f"## {query_name}\n\n```sql\n{query_text.strip()}\n```\n\n")
        output_file.write(query_results[query_name].to_string(index=False))
        output_file.write("\n\n")

    output_file.write("## 5. JOIN query\n\n```sql\n")
    output_file.write(join_query.strip())
    output_file.write("\n```\n\n### SQLite result\n\n")
    output_file.write(sql_join_result.to_string(index=False))
    output_file.write("\n\n### pandas merge result\n\n")
    output_file.write(merge_join_result.to_string(index=False))
    output_file.write(
        f"\n\n**Results match:** {sql_join_result.equals(merge_join_result)}\n"
    )

connection.close()