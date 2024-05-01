import os
import re
import pickle
import pandas as pd
import torch
import torch.nn as nn

import torchtext
from torchtext.functional import to_tensor


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_documents(content_path):
    df_content = pd.read_csv(content_path, sep=';')

    paper_ids = df_content["paper_id"].values

    titles = [str(string) for string in list(df_content["title"].values)]

    # filtered_df = df_content[df_content['keywords'].notna()]
    filtered_df = df_content.fillna("NULL")
    keywords = [' '.join(re.split(r'[;\t]', keywords)) for keywords in filtered_df["keywords"].values]

    abstracts = [str(string) for string in list(df_content["abstract"].values)]
    
    return paper_ids, titles, keywords, abstracts


class AIModel:
    def __init__(self, content_path: str) -> None:
        self.df_content = pd.read_csv(content_path, sep=';')

        xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
        self.model = xlmr_large.get_model().to(DEVICE)
        self.transform = xlmr_large.transform()
        self.model.eval()
        
        if os.path.exists("cache/AIModel.pkl"):
            with open("cache/AIModel.pkl", 'rb') as file:
                cache = pickle.load(file)

                self.paper_ids = cache["paper_ids"]
                self.title_features = cache["title_features"]
                self.keyword_features = cache["keyword_features"]
                self.abstract_features = cache["abstract_features"]
        else:
            self.paper_ids, titles, keywords, abstracts = get_documents(content_path)

            batch_size = 32

            title_features = []
            for i in range(0, titles.__len__(), batch_size):
                x_batch = to_tensor(self.transform(titles[i:i+batch_size]), padding_value=1).to(DEVICE)
                with torch.no_grad():
                    features = torch.mean(self.model(x_batch), dim=1)
                    title_features.append(features.to(torch.float16).to("cpu"))
            self.title_features = torch.concat(title_features, dim=0)

            keyword_features = []
            for i in range(0, keywords.__len__(), batch_size):
                x_batch = to_tensor(self.transform(keywords[i:i+batch_size]), padding_value=1).to(DEVICE)
                with torch.no_grad():
                    features = torch.mean(self.model(x_batch), dim=1)
                    keyword_features.append(features.to(torch.float16).to("cpu"))
            self.keyword_features = torch.concat(keyword_features, dim=0)

            abstract_features = []
            for i in range(0, abstracts.__len__(), batch_size):
                x_batch = to_tensor(self.transform(abstracts[i:i+batch_size]), padding_value=1).to(DEVICE)
                with torch.no_grad():
                    features = torch.mean(self.model(x_batch), dim=1)
                    abstract_features.append(features.to(torch.float16).to("cpu"))
            self.abstract_features = torch.concat(abstract_features, dim=0)

            with open("cache/AIModel.pkl", 'wb') as file:
                pickle.dump({
                    "paper_ids": self.paper_ids, "title_features": self.title_features, "keyword_features": self.keyword_features, "abstract_features": self.abstract_features, }, file)
    

    def compute_cossim(self, query, schema: str):
        if schema == "title":
            doc_features = self.title_features.to(torch.float32)
        elif schema == "keywords":
            doc_features = self.keyword_features.to(torch.float32)
        elif schema == "abstract":
            doc_features = self.abstract_features.to(torch.float32)
        else:
            raise ValueError("`{}` is an invalid value for parameter `schema`.")
        
        self.model.eval()
        x_batch = to_tensor(self.transform([query]), padding_value=1).to(DEVICE)
        with torch.no_grad():
            query_feature = torch.mean(self.model(x_batch), dim=1).to("cpu")

        norm_query = torch.sqrt(torch.sum(torch.square(query_feature), dim=1))
        norm_doc = torch.sqrt(torch.sum(torch.square(doc_features), axis=1))
        return torch.sum(query_feature * doc_features, axis=1) / norm_query / norm_doc
    
    def compute_ranking(self, query, schema: str, threshold=50):
        cossim = self.compute_cossim(query, schema)
        ascending_indices = torch.argsort(cossim)
        descending_indices = torch.flip(ascending_indices, dims=[0])

        paper_ids = self.paper_ids[descending_indices[:threshold]]
        filtered_df = self.df_content[self.df_content['paper_id'].isin(paper_ids)]
        titles = filtered_df['title'].tolist()
        keywords = filtered_df['keywords'].tolist()
        abstracts = filtered_df['abstract'].tolist()
        
        return titles, keywords, abstracts


if __name__ == "__main__":
    model = AIModel(content_path="src/Content.csv")

    paper_ids, titles, keywords, abstracts = get_documents(content_path="src/Content.csv")
    print(titles[2])
    titles, keywords, abstracts = model.compute_ranking(titles[2], schema="title", threshold=50)
    print(titles)