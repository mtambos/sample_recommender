# Based on https://github.com/groveco/content-engine/blob/master/engines.py
# Original Author: Chris Clark https://github.com/chrisclark
# @author: Mario Tambos https://github.com/mtambos

import pickle

from flask import current_app
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time


def info(msg):
    current_app.logger.info(msg)


class ContentEngine(object):
    def train(self, data_source):
        start = time.time()
        ds = pd.read_csv(data_source)
        info("Training data ingested in %s seconds." % (time.time() - start))

        start = time.time()
        self._train(ds)
        info("Engine trained in %s seconds." % (time.time() - start))

    @classmethod
    def _train(cls, ds):
        """
        Train the engine.

        Create a TF-IDF matrix of unigrams, bigrams, and trigrams
        for each product. The 'stop_words' param tells the TF-IDF
        module to ignore common english words like 'the', etc.

        :param ds: A pandas dataset containing two fields: description & id
        :return: Nothing!
        """

        tf = TfidfVectorizer(analyzer='word',
                             ngram_range=(1, 3),
                             min_df=0,
                             stop_words='english')
        tfidf_matrix = tf.fit_transform(ds['description'])
        with open('ds.pickle', 'wb') as fp:
            pickle.dump(ds, fp)

        with open('tfidf_matrix.pickle', 'wb') as fp:
            pickle.dump(tfidf_matrix, fp)

        with open('tf_model.pickle', 'wb') as fp:
            pickle.dump(tf, fp)

    @classmethod
    def recommend(cls, content, num):
        """
        Couldn't be simpler! Just retrieves the similar items and
        their 'score' from redis.

        :param content: string
        :param num: number of similar items to return
        :return: A list of lists like: [[0.2203, "[doc2]"],
        [0.1693, "[doc2]"], ...]. The first item in each sub-list is
        the similarity score and the second the recommended document. Sorted
        by similarity score, descending.
        """
        with open('ds.pickle', 'rb') as fp:
            ds: pd.DataFrame = pickle.load(fp)

        with open('tfidf_matrix.pickle', 'rb') as fp:
            tfidf_matrix: spmatrix = pickle.load(fp)

        with open('tf_model.pickle', 'rb') as fp:
            tf: TfidfVectorizer = pickle.load(fp)

        tfidf_content = tf.transform([content])
        cosine_similarities = linear_kernel(tfidf_content, tfidf_matrix)[0]

        similar_indices = cosine_similarities.argsort()[:-num-1:-1]
        similar_items = [(cosine_similarities[i], ds['description'].iloc[i])
                         for i in similar_indices]
        return similar_items

content_engine = ContentEngine()
