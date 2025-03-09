import os
import warnings
import numpy as np
import pandas as pd
import pickle
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')  # Suppress warnings for clarity

def load_text_data(text_path):
    """ Load cleaned text data from CSV files. """
    titles = pd.read_csv(os.path.join(text_path, 'titles_clean.csv'))['Text'].tolist()
    descriptions = pd.read_csv(os.path.join(text_path, 'des_clean.csv'))['Text'].tolist()
    body = pd.read_csv(os.path.join(text_path, 'news_electric_clean.csv'))['Text'].tolist()
    return titles, descriptions, body


def jaccard_similarity(topic_1, topic_2):
    """ Compute Jaccard similarity between two topics. """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))
    return len(intersection) / len(union)


def build_dictionary(docs):
    """ Build a dictionary and corpus from tokenized documents. """
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.2)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return corpus, dictionary.id2token, dictionary


def train_lda_model(i, corpus, id2word):
    """ Train an LDA model with the given number of topics. """
    print(f"Training LDA model with {i} topics...")
    lda_model = LdaMulticore(corpus=corpus, id2word=id2word, alpha='asymmetric',
                             num_topics=i, passes=20, random_state=42, workers=7)
    topics = [[word[0] for word in topic[1]] for topic in lda_model.show_topics(num_topics=i, num_words=15, formatted=False)]
    return lda_model, topics


def compute_lda_metrics(num_topics, lda_topics, lda_models, docs, dictionary):
    """ Compute coherence and stability metrics for LDA models. """
    lda_stability = {i: [jaccard_similarity(topic1, topic2) for topic1 in lda_topics[i] for topic2 in lda_topics[i + 1]]
                     for i in range(len(num_topics) - 1)}
    mean_stabilities = [np.mean(lda_stability[i]) for i in num_topics[:-1]]
    coherences = [CoherenceModel(model=lda_models[i], texts=docs, dictionary=dictionary, coherence='c_v').get_coherence()
                  for i in num_topics[:-1]]
    coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(len(num_topics) - 1)]
    return mean_stabilities, coherences, coh_sta_diffs


if __name__ == '__main__':
    print("Loading data...")
    data_path = 'your path'
    save_path = 'your path'
    os.makedirs(save_path, exist_ok=True)

    titles, descriptions, body = load_text_data(data_path)
    
    docs = [s.strip().split() for s in body if isinstance(s, str)]
    corpus, id2word, dictionary = build_dictionary(docs)
    
    # Save dictionary and corpus
    with open(os.path.join(save_path, 'corpus.pkl'), 'wb') as f:
        pickle.dump(corpus, f)
    with open(os.path.join(save_path, 'id2word.pkl'), 'wb') as f:
        pickle.dump(id2word, f)
    with open(os.path.join(save_path, 'dictionary.pkl'), 'wb') as f:
        pickle.dump(dictionary, f)
    
    print("Training LDA models...")
    num_topics = list(range(1, 101))
    num_cores = os.cpu_count() if os.cpu_count() else 6
    results = Parallel(n_jobs=num_cores)(delayed(train_lda_model)(i, corpus, id2word) for i in num_topics)
    
    with open(os.path.join(save_path, 'lda_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("LDA training completed!")
