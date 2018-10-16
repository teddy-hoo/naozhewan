import string
import collections
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import pandas as pd


def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    text = text.translate(string.punctuation)
    tokens = word_tokenize(text, language='chinese')
    print(tokens)
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def cluster_texts(texts, clusters=3):
    """
    Transform texts to Tf-Idf coordinates and cluster texts using K-Means
    使用tf-idf对文本打分  然后使用k-means进行聚类
    """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 max_df=0.5, min_df=0.0, lowercase=True)
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    return clustering


if __name__ == "__main__":
    data = pd.read_csv('../../data/train/1.csv')
    text_value = data['许可经营项目'].values[:10]
    texts = []
    for t in text_value:
        if t != t:
            texts.append('')
        else:
            print(t)
            texts.append(t.replace('（依法须经批准的项目，经相关部门批准后方可开展经营活动）', ''))
    clusters = cluster_texts(texts, 2)
    pprint(dict(clusters)[0])
    # pprint(dict(clusters)[1])
    # pprint(dict(clusters)[2])
    # pprint(dict(clusters)[3])
    # pprint(dict(clusters)[4])
    # pprint(dict(clusters)[5])
    # pprint(dict(clusters)[6])
