import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle

def lambda_handler(event='sad', context=None,url=""):
    '''
    감정이 들어왔다고 해야해용
    '''

    # emotion 들어온거에 맞게 데이터를 받아오자
    with open(f'/var/task/{event}0916_embeddings.pkl', 'rb') as file:
        embedding = pickle.load(file)
    df=pd.read_csv(f'/var/task/{event}0916_embedding.csv')
    embedder = SentenceTransformer("/var/task/model_2")

    #
    queries = [
        '오늘 학교에서 혜진이랑 싸웠어',
        '진짜 너무 싫어',
        '개빡치네 진짜'
    ]


    for id, query in enumerate(queries):

        query_embedding = embedder.encode(query, convert_to_tensor=True)
        max_cos = 0
        musi = ""
        singer = ""
        cover = ""
        youtube = ""

        for idx in range(len(embedding)):
            cos_scores = util.pytorch_cos_sim(query_embedding, embedding[idx])[0]
            cos_scores = cos_scores.cpu()
            n_cos_scores = cos_scores.numpy().reshape((cos_scores.shape[0], 1))

            if max_cos < np.max(n_cos_scores):
                max_cos = np.max(n_cos_scores)
                musi = df['song'][idx]
                singer = df['singer'][idx]
                cover = df['cover'][idx]
                youtube = df['youtube'][idx]

    return {
        'Song':musi,
        'Singer':singer,
        'cover':cover,
        'youtube':youtube
    }
