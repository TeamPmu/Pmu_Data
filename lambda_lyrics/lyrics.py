import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import kss

def lambda_handler(event, context=None, url=""):
    '''
    event -> dict sample : {'emotion' : ex'sad' , 'text' : 'i hate you'}
    '''
    emotion = event.get('emothion')


    # emotion 들어온거에 맞게 데이터를 받아오자
    with open(f'/var/task/{emotion}_prototype_embeddings.pkl', 'rb') as file:
        embedding = pickle.load(file)
    df = pd.read_csv(f'/var/task/{emotion}_prototype_embedding.csv')
    embedder = SentenceTransformer("/var/task/model_2")


    top_k = []

    sentencess = []
    for sent in kss.split_sentences(event['text']):
        sentencess.append(sent)
    for id, query in enumerate(sentencess):

        query_embedding = embedder.encode(query, convert_to_tensor=True)
        for idx in range(len(embedding)):
            cos_scores = util.pytorch_cos_sim(query_embedding, embedding[idx])[0]
            cos_scores = cos_scores.cpu()
            n_cos_scores = cos_scores.numpy().reshape((cos_scores.shape[0], 1))

            top_k.append([np.max(n_cos_scores), idx])

    extract = ['song', 'singer', 'cover', 'youtube']
    top_k = [item[1] for item in (sorted(top_k, reverse=True)[:5])]
    top_r = df.loc[top_k, extract]

    return {
        'Song': list(top_r['song']),
        'Singer': list(top_r['singer']),
        'cover': list(top_r['cover']),
        'youtube': list(top_r['youtube'])
    }
