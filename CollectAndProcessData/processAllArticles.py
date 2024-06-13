"""
This script summerize and generate embedings for each article.
"""
import pickle
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

#model
sentence_similarity_model = "paraphrase-MiniLM-L6-v2"

#Read data
#data_path = "/Users/falbanese/Documents/Projects/PolarizationCHP/CollectAndProcessData/Data/articles_23FEB_summery.csv" #summeries were made in a Notebook due to model size https://colab.research.google.com/drive/1YKDBW9q4qJcaLPnecTn59wDctixY4G0a?usp=sharing
data_path = "/Users/falbanese/Documents/OtherProjects/PolarizationCHP/CollectAndProcessData/Data/articles_3JUN.csv" 

data = pd.read_csv(data_path, sep=";")

#Generate Embeddings
model = SentenceTransformer(sentence_similarity_model)

article_embeddings = []
for article_title, article_text in tqdm(zip(data.article_title, data.article_text)):
    text = f"Title: {article_title} Text:{article_text}" if article_title is not None else article_text
    article_embeddings.append(model.encode(text))
data["article_embedding"] = article_embeddings

summery_embeddings = []
for article_summery in tqdm(data.article_title): #tqdm(data.article_summery): There is no summery for 3JUN dataset
    summery_embeddings.append(model.encode(article_summery))
data["summery_embedding"] = summery_embeddings

#Save data
output_path = data_path.replace(".csv", "") + "_processed.pickle"
with open(output_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
"""
to measure the distance use:

from scipy.spatial import distance

def cosine_similarity(vec1, vec2):
    return 1-distance.cosine(vec1, vec2)

similarity = cosine_similarity(embedding1, embedding2)[0]
"""