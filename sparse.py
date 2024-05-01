import os
import sqlite3
import pandas as pd
import nltk
from wordcloud import WordCloud, STOPWORDS  
from nltk.corpus import wordnet as wn
from matplotlib import pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')

DB_NAME = 'src/papers.sqlite'


def load_sqlite_as_csv(dir_path):

    # Database file path and table name
    db_path = dir_path + "/papers.sqlite"  # Update this to the path of your SQLite database

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

    # Load the table into a pandas DataFrame, then save DataFrame to CSV
    df = pd.read_sql_query(f"SELECT * FROM Papers", conn)
    csv_path = dir_path + "/Papers.csv"  # Update this to your desired CSV path
    df.to_csv(csv_path, sep=";", index=False)

    df = pd.read_sql_query(f"SELECT * FROM Content", conn)
    csv_path = dir_path + "/Content.csv"  # Update this to your desired CSV path
    df.to_csv(csv_path, sep=";", index=False)


    # Close the connection
    conn.close()

def gen_keywords():
    # Connect to your SQLite database
    conn = sqlite3.connect(DB_NAME)

    # Write your SQL query
    query = "SELECT * FROM Content"

    # Load the query result into a pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Concat keywords.
    keywords = []
    for string in df['keywords']:
        keywords = keywords + string.split('\t')

    # Close the SQLite connection
    conn.close()

    return ' '.join(keywords)

def gen_titles():
    # Connect to your SQLite database
    conn = sqlite3.connect(DB_NAME)

    # Write your SQL query
    query = "SELECT * FROM Content"

    # Load the query result into a pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Concat titles.
    titles = [string for string in df['title']]

    # Close the SQLite connection
    conn.close()

    return ' '.join(titles)

def gen_abstracts():
    # Connect to your SQLite database
    conn = sqlite3.connect(DB_NAME)

    # Write your SQL query
    query = "SELECT * FROM Content"

    # Load the query result into a pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Concat titles.
    abstracts = [string for string in df['abstract']]

    # Close the SQLite connection
    conn.close()

    return ' '.join(abstracts)

def gen_wordcloud(text, schema):
    wordcloud = WordCloud(
        width=800, height=800, background_color='black', stopwords=set(STOPWORDS),
        min_font_size=20, max_words=100).generate(text)
    # Plot the WordCloud image
    plt.figure(figsize=(8, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title('Top 100 Words in the Titles of Papers')
    plt.savefig(f'static/{schema}_wordcloud.png')  # Save the word cloud image with the schema name


if __name__ == '__main__':
    # load_sqlite_as_csv(dir_path="/home/lizheng/Documents/CSCI 626/project/src")

    gen_wordcloud(text=gen_titles(), schema="title")
    gen_wordcloud(text=gen_keywords(), schema="keywords")
    gen_wordcloud(text=gen_abstracts(), schema="abstract")
