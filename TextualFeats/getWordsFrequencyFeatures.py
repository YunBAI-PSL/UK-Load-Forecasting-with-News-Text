import os
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from get_topics import build_dictionary


def load_lda_data(path_lda, text_type):
    """ Load LDA model data (corpus, dictionary, id2word). """
    base_path = os.path.join(path_lda, f'BBC_LDA_{text_type}')
    
    with open(os.path.join(base_path, 'corpus.pkl'), 'rb') as f:
        corpus = pickle.load(f)
    with open(os.path.join(base_path, 'dictionary.pkl'), 'rb') as f:
        dictionary = pickle.load(f)
    with open(os.path.join(base_path, 'id2word.pkl'), 'rb') as f:
        id2word = pickle.load(f)
    
    return corpus, dictionary, id2word


def get_dates_from_corpus(df_path, text_type):
    """ Retrieve dates for each corpus entry. """
    df = pd.read_csv(os.path.join(df_path, f'{text_type}_clean.csv')).dropna()
    return df['Date'].tolist()


def get_high_frequency_words(dictionary, min_freq):
    """ Retrieve words appearing more than min_freq times. """
    word_freq = np.array(list(dictionary.dfs.values()))
    word_ids = np.array(list(dictionary.dfs.keys()))
    return list(word_ids[word_freq > min_freq])


def generate_word_frequency(df_path, text_type, high_freq_words, corpus, id2word, output_file):
    """ Generate word frequency dataframe. """
    dates = get_dates_from_corpus(df_path, text_type)
    word_freq_df = pd.DataFrame({'Date': dates})
    
    for word_id in high_freq_words:
        word = id2word[word_id]
        word_freq_df[word] = [sum(wt[1] for wt in doc if wt[0] == word_id) for doc in corpus]
    
    word_freq_df.groupby('Date').sum().to_csv(output_file, index=True)


def calculate_corpus_volume(corpus):
    """ Compute total number of words in corpus. """
    return sum(len(doc) for doc in corpus)


# Set paths
feat_path = 'your path'
df_path = 'your path'
text_type = 'body' # or 'titles' or 'des'

# Load LDA data
corpus, dictionary, id2word = load_lda_data(feat_path, text_type)
calculate_corpus_volume(corpus)

# Define frequency threshold
min_freq_dict = {'titles': 200, 'des': 400, 'body': 5000, 'electricity': 100}
min_freq = min_freq_dict.get(text_type, 5000)

# Get high frequency words
high_freq_words = get_high_frequency_words(dictionary, min_freq)

# Load cleaned data
body_df = pd.read_csv(os.path.join(df_path, 'body_clean.csv'))
body_df['Date'] = pd.to_datetime(body_df['Date'])

# Aggregate text per date
aggregated_df = body_df.groupby('Date')['Text'].agg(lambda x: ' '.join(x)).reset_index()
body_docs = [s.strip().split() for s in aggregated_df.Text.dropna() if isinstance(s, str)]

# Generate word frequency dataframe
output_path = 'word_freq.csv'
generate_word_frequency(df_path, text_type, high_freq_words, corpus, id2word, output_path)




