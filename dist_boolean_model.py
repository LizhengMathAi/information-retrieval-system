import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/home/zhengli/spark-3.1.1-bin-hadoop3.2"

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True) # Property used to format output tables better

import re
import pyspark.sql.functions as F
import nltk
from nltk.corpus import stopwords
import numpy as np

import argparse


def string_to_ngram_set(string, n):
    """
    Convert a context into n-gram set.

    Parameters:
        string (str): Input string.
        n (int): Size of n-grams.

    Returns:
        set: Set of n-grams in the context.
    """
    # Tokenize the string into phrases
    phrases = re.split(r'[,.;\t]', string.lower())

    # Tokenize each phrase into words
    tokens = [phrase.split() for phrase in phrases]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [[word.lower() for word in items if word.lower() not in stop_words] for items in tokens]
    
    # Word lemmatization
    wnl = nltk.WordNetLemmatizer()
    tokens = [[wnl.lemmatize(word) for word in items] for items in tokens]
    # print(tokens)
    # raise ValueError

    # Generate n-grams
    ngrams = []
    for token in tokens:
        for i in range(len(token) - n + 1):
            ngrams.append(' '.join(token[i:i+n]))

    # Convert n-grams list to a set
    ngram_set = set(ngrams)

    return ngram_set


# Define the function to compute inner similarity between two n-gram sets
def inner_similarity(set1, set2):
    """
    Compute the cosine similarity coefficient between two n-gram sets.

    Parameters:
        set1 (set): First n-gram set.
        set2 (set): Second n-gram set.

    Returns:
        float: Cosine similarity coefficient.
    """
    # Union of n-gram sets to get the vocabulary
    vocabulary = set1.union(set2)

    # Convert n-gram sets to binary vectors
    vector1 = np.array([1 if ngram in set1 else 0 for ngram in vocabulary])
    vector2 = np.array([1 if ngram in set2 else 0 for ngram in vocabulary])

    # Compute inner similarity coefficient
    inner_sim = np.dot(vector1, vector2)

    return inner_sim


def compute_ranking(df, query, schema: str, n:str, threshold=50):
    query_ngram_set = string_to_ngram_set(query, n)
    # Define the example n-gram set to compare with each n-gram set in ngram_sets
    query_ngram_set_broadcast = spark.sparkContext.broadcast(query_ngram_set)

    # Apply the function to generate n-gram sets for each row in the specific column
    filtered_df = df.filter(F.col(schema).isNotNull() & (~F.isnan(schema)))
    # filtered_df.show(5, truncate=True, vertical=True)
    if schema == "title":
        ngram_sets = filtered_df.rdd.map(lambda row: string_to_ngram_set(row.title, n))
    elif schema == "keywords":
        ngram_sets = filtered_df.rdd.map(lambda row: string_to_ngram_set(row.keywords, n))
    elif schema == "abstract":
        ngram_sets = filtered_df.rdd.map(lambda row: string_to_ngram_set(row.abstract, n))
    else:
        raise ValueError("`schema` should be one if ['title', 'keywords', 'abstract'].")
    # Zip RDD elements with their indices
    indexed_ngram_sets = ngram_sets.zipWithIndex()

    # Compute inner similarity between query_ngram_set and each ngram_set in ngram_sets
    inner_similarities_with_index = indexed_ngram_sets.map(lambda x: (x[1], inner_similarity(query_ngram_set_broadcast.value, x[0])))
    # Sort the results by inner similarity from high to low
    sorted_indices = inner_similarities_with_index.sortBy(lambda x: x[1], ascending=False)
    # Collect the sorted indices
    sorted_indices_list = sorted_indices.collect()

    # Collect the DataFrame into a list of rows
    rows = filtered_df.collect()

    # Retrieve data for each index in the sorted list of indices
    paper_ids, titles, keywords, abstracts = [], [], [], []
    paper_ids = [rows[idx].paper_id for idx, _ in sorted_indices_list[:threshold]]
    titles = [rows[idx].title for idx, _ in sorted_indices_list[:threshold]]
    keywords = [rows[idx].keywords for idx, _ in sorted_indices_list[:threshold]]
    abstracts = [rows[idx].abstract for idx, _ in sorted_indices_list[:threshold]]
    return paper_ids, titles, keywords, abstracts



class DistBooleanModel:
    def __init__(self, content_path: str) -> None:
        self.df_content = spark.read.csv(content_path, header=True, inferSchema=True, sep=';')
    
    def compute_ranking(self, query, schema: str, n=2, threshold=50):
        return compute_ranking(self.df_content, query, schema, n, threshold)


if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser(description='Run PySpark text processing.')
    parser.add_argument('--path', default="/home/zhengli/Documents/CSCI-626/project/src/Content.csv", type=str, help='The path of `Content.csv`.')
    parser.add_argument('--query', default="artificial neural networks;\tdiffusion model;knowledge;\tby Large Language Model", type=str, help='The query string to process.')
    parser.add_argument('--schema', default="abstract", type=str, help='The schema to searching.')
    parser.add_argument('--n', default=2, type=int, help='The size of n-grams.')
    args = parser.parse_args()

    # Load DataFrame
    df = spark.read.csv(args.path, header=True, inferSchema=True, sep=';')
    
    # Use arguments from argparse
    query_ngram_set = string_to_ngram_set(args.query, args.n)
    # print(f"N-Gram={args.n}: {query_ngram_set}")
    # print("Ranking:")
    paper_ids, titles, keywords, abstracts = compute_ranking(df, args.query, args.schema, args.n)
    for idx, t, k in zip(paper_ids, titles, keywords):
        print('[{}]\n{}\n[{}]\n'.format(idx, t, k))

