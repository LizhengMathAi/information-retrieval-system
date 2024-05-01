import os
import re
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords


def get_documents(content_path):
    df_content = pd.read_csv(content_path, sep=';')

    paper_ids = df_content["paper_id"].values

    titles = df_content["title"].values

    # filtered_df = df_content[df_content['keywords'].notna()]
    filtered_df = df_content.fillna("NULL")
    keywords = np.array([' '.join(re.split(r'[;\t]', keywords)) for keywords in filtered_df["keywords"].values])

    abstracts = df_content["abstract"].values
    
    return paper_ids, titles, keywords, abstracts


def tokenization(documents):
    tokens_list, types_list = [], []
    for doc in documents:
        tokens = nltk.word_tokenize(str(doc))
        tokens_list.append(tokens)

        types = list(set(tokens))
        types_list.append(types)

    return tokens_list, types_list


def remove_stopword(tokens, lower=True):
    stop_words = set(stopwords.words('english'))
    if lower:
        return [word for word in tokens if word.lower() not in stop_words]
    else:
        return [word for word in tokens if word not in stop_words]


def lemmatization(tokens, lower=True, unique=True):
    wnl = nltk.WordNetLemmatizer()

    if lower:
        tokens = [t.lower() for t in tokens]
    
    tokens = [wnl.lemmatize(t) for t in tokens]

    if unique:
        # return sorted(list(set(tokens)))
        return set(tokens)
    else:
        return tokens


def compute_tfidf(documents):
    # Get tokens and types.
    tokens_list, types_list = tokenization(documents)
    entire_tokens, entire_types = tokenization([" ".join([str(string) for string in documents]), ])
    entire_tokens, entire_types = entire_tokens[0], entire_types[0]

    # Remove stopwords.
    types_list = [remove_stopword(types, lower=True) for types in types_list]
    entire_types = remove_stopword(entire_types, lower=True)

    # Lemmatizate types.
    terms_list = [lemmatization(types, lower=True, unique=True) for types in types_list]
    entire_terms = lemmatization(entire_types, lower=True, unique=True)

    # Compute the TF
    tokens_list = [remove_stopword(tokens, lower=True) for tokens in tokens_list]
    tokens_list = [lemmatization(tokens, lower=True, unique=False) for tokens in tokens_list]
    tf = np.array([[tokens.count(t) for t in entire_terms] for tokens in tokens_list])
    tf = tf / np.max(tf, axis=1, keepdims=True)

    # Compute the IDF
    idf = np.log2(documents.__len__() / np.sum(tf > 0, axis=0))

    # Compute the TF-IDF
    tfidf = tf * idf[None, :]

    return entire_terms, idf, tfidf


class VectorModel:
    def __init__(self, content_path: str) -> None:

        self.df_content = pd.read_csv(content_path, sep=';')
        
        if os.path.exists("cache/VectorModel.pkl"):
            with open("cache/VectorModel.pkl", 'rb') as file:
                cache = pickle.load(file)

                self.paper_ids = cache["paper_ids"]

                self.title_terms = cache["title_terms"]
                self.title_idf = cache["title_idf"]
                self.title_tfidf = cache["title_tfidf"]

                self.keyword_terms = cache["keyword_terms"]
                self.keyword_idf = cache["keyword_idf"]
                self.keyword_tfidf = cache["keyword_tfidf"]

                self.abstract_terms = cache["abstract_terms"]
                self.abstract_idf = cache["abstract_idf"]
                self.abstract_tfidf = cache["abstract_tfidf"]
        else:
            self.paper_ids, titles, keywords, abstracts = get_documents(content_path)
            self.title_terms, self.title_idf, self.title_tfidf = compute_tfidf(titles)
            self.keyword_terms, self.keyword_idf, self.keyword_tfidf = compute_tfidf(keywords)
            self.abstract_terms, self.abstract_idf, self.abstract_tfidf = compute_tfidf(abstracts)

            with open("cache/VectorModel.pkl", 'wb') as file:
                pickle.dump({
                    "paper_ids": self.paper_ids, 
                    "title_terms": self.title_terms, "title_idf": self.title_idf, "title_tfidf": self.title_tfidf, 
                    "keyword_terms": self.keyword_terms, "keyword_idf": self.keyword_idf, "keyword_tfidf": self.keyword_tfidf, 
                    "abstract_terms": self.abstract_terms, "abstract_idf": self.abstract_idf, "abstract_tfidf": self.abstract_tfidf}, file)

    def compute_tfidf(self, query, schema: str):
        if schema == "title":
            terms = self.title_terms
            idf = self.title_idf
        elif schema == "keywords":
            terms = self.keyword_terms
            idf = self.keyword_idf
        elif schema == "abstract":
            terms = self.abstract_terms
            idf = self.abstract_idf
        else:
            raise ValueError("`{}` is an invalid value for parameter `schema`.")
        
        # Get tokens and types.
        tokens_list, types_list = tokenization([query])

        # Compute the TF
        tokens_list = [remove_stopword(tokens, lower=True) for tokens in tokens_list]
        tokens_list = [lemmatization(tokens, lower=True, unique=False) for tokens in tokens_list]
        tf = np.array([[tokens.count(t) for t in terms] for tokens in tokens_list])
        tf = tf / np.max(tf, axis=1, keepdims=True)

        # Compute the TF-IDF
        print(idf)
        tfidf = tf * idf[None, :]

        return tfidf[0]
    
    def compute_cossim(self, query, schema: str):
        if schema == "title":
            doc_tfidf = self.title_tfidf
        elif schema == "keywords":
            doc_tfidf = self.keyword_tfidf
        elif schema == "abstract":
            doc_tfidf = self.abstract_tfidf
        else:
            raise ValueError("`{}` is an invalid value for parameter `schema`.")

        query_tfidf = np.nan_to_num(self.compute_tfidf(query, schema), nan=0)
        norm_query = np.sqrt(np.sum(np.square(query_tfidf)))
        doc_tfidf = np.nan_to_num(doc_tfidf, nan=0)
        norm_doc = np.sqrt(np.sum(np.square(doc_tfidf), axis=1))
        return np.sum(query_tfidf[None, :] * doc_tfidf, axis=1) / norm_query / norm_doc
    
    def compute_ranking(self, query, schema: str, threshold=50):
        cossim = self.compute_cossim(query, schema)
        ranking_indices = np.argsort(cossim)[:threshold]

        paper_ids = self.paper_ids[ranking_indices]
        filtered_df = self.df_content[self.df_content['paper_id'].isin(paper_ids)]
        titles = filtered_df['title'].tolist()
        keywords = filtered_df['keywords'].tolist()
        abstracts = filtered_df['abstract'].tolist()
        
        return titles, keywords, abstracts


if __name__ == "__main__":
    model = VectorModel(content_path="src/Content.csv")

    paper_ids, titles, keywords, abstracts = get_documents(content_path="src/Content.csv")
    titles, keywords, abstracts = model.compute_ranking(abstracts[2], schema="abstract", threshold=50)
    print(titles)