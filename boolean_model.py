import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
wnl = nltk.WordNetLemmatizer()


def get_documents(content_path):
    df_content = pd.read_csv(content_path, sep=';')

    paper_ids = df_content["paper_id"].values

    titles = [str(string) for string in list(df_content["title"].values)]

    # filtered_df = df_content[df_content['keywords'].notna()]
    filtered_df = df_content.fillna("NULL")
    keywords = [' '.join(re.split(r'[;\t]', keywords)) for keywords in filtered_df["keywords"].values]

    abstracts = [str(string) for string in list(df_content["abstract"].values)]
    
    return paper_ids, titles, keywords, abstracts


def context_to_ngram_set(context, n):
    """
    Convert a context into n-gram set.

    Parameters:
        context (str): Input context.
        n (int): Size of n-grams.

    Returns:
        set: Set of n-grams in the context.
    """
    # Tokenize the context into words
    tokens = nltk.word_tokenize(context)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    tokens = [wnl.lemmatize(t) for t in tokens]

    # Generate n-grams
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    # Convert n-grams list to a set
    ngram_set = set(ngrams)

    return ngram_set


def inner_similarity_ngram_set_mode(set1, set2):
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


def cosine_similarity_ngram_set_mode(set1, set2):
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

    # Compute cosine similarity coefficient
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0  # Return 0 if one of the vectors is a zero vector
    cosine_sim = np.dot(vector1, vector2) / (norm_vector1 * norm_vector2)

    return cosine_sim


class BooleanModel:
    def __init__(self, content_path: str) -> None:
        self.df_content = pd.read_csv(content_path, sep=';')
        self.paper_ids, self.titles, self.keywords, self.abstracts = get_documents(content_path)

    def compute_sim(self, query, schema: str, n: int):
        if schema == "title":
            documents = self.titles
        elif schema == "keywords":
            documents = self.keywords
        elif schema == "abstract":
            documents = self.abstracts
        else:
            raise ValueError("`{}` is an invalid value for parameter `schema`.")
        
        inner_sim = []
        set1 = context_to_ngram_set(query, n)
        for context in documents:
            set2 = context_to_ngram_set(context, n)
            inner_sim.append(cosine_similarity_ngram_set_mode(set1, set2))
        return np.array(inner_sim)
    
    def compute_ranking(self, query, schema: str, n=2, threshold=50):
        sim = self.compute_sim(query, schema, n)
        ranking_indices = np.argsort(sim)[::-1]

        paper_ids = self.paper_ids[ranking_indices[:threshold]]
        filtered_df = self.df_content[self.df_content['paper_id'].isin(paper_ids)]
        titles = filtered_df['title'].tolist()
        keywords = filtered_df['keywords'].tolist()
        abstracts = filtered_df['abstract'].tolist()
        
        return paper_ids, titles, keywords, abstracts
    

if __name__ == "__main__":
    model = BooleanModel(content_path="src/Content.csv")

    query = "Quality-Diversity through Massive"
    print(query)
    paper_ids, titles, keywords, abstracts = model.compute_ranking(query, schema="title", threshold=50)
    print(titles)