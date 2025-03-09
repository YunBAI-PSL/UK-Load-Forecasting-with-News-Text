import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

def plot_num_topics(df_name, title, path_lda):
    """Plot the number of topics based on ATO and TC metrics."""
    topic_index_df = pd.read_csv(os.path.join(path_lda, f'Num_of_topics_{df_name}.csv'))
    num_topics = [i + 2 for i in range(len(topic_index_df) + 5)]
    
    x_ato, x_tc = topic_index_df['ATO'], topic_index_df['TC']
    x_diff = (x_tc - x_ato).tolist()
    ideal_num = x_diff.index(max(x_diff)) + 2
    
    print(f'The ideal number of topics is: {ideal_num}')
    
    plt.plot(num_topics, x_ato.tolist() + [None] * 5, label='Average Topic Overlap (ATO)', linewidth=3)
    plt.plot(num_topics, x_tc.tolist() + [None] * 5, label='Topic Coherence (TC)', linewidth=3)
    plt.axvline(x=ideal_num, label='Ideal number of topics', color='#2483AA', linewidth=3)
    plt.axvspan(xmin=ideal_num - 1, xmax=ideal_num + 1, alpha=0.5, facecolor='#cdffbf')
    
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylim([0, max(max(x_ato), max(x_tc), max(x_diff)) * 1.1])
    plt.xlim([1, num_topics[-1] - 1])
    plt.title(title, fontsize=30, loc='left')
    plt.ylabel('Metric values', fontsize=30)
    if title == 'E':
        plt.xlabel('Number of topics', fontsize=30)
    if title == 'T':
        plt.legend(fontsize=30, loc='lower center')

def plot_num_of_topics_all(path_lda):
    """Plot topic numbers for all datasets."""
    gs = gridspec.GridSpec(4, 1)
    fig = plt.figure(figsize=(18, 20))
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    
    for i, (df_name, title) in enumerate(zip(['titles', 'des', 'body', 'electricity'], ['T', 'D', 'B', 'E'])):
        ax = fig.add_subplot(gs[i, 0])
        plot_num_topics(df_name, title, path_lda)
        if i < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
    

def topics_features(df_name, path_lda, df_path, num_topics): 
    """Generate topic distribution features."""
    with open(os.path.join(path_lda, f'BBC_LDA_{df_name}/corpus.pkl'), 'rb') as f1:
        corpus = pickle.load(f1)
    with open(os.path.join(path_lda, f'BBC_LDA_{df_name}/LDA_results_all.pkl'), 'rb') as f2:
        lda_results_all = pickle.load(f2)
    with open(os.path.join(path_lda, f'BBC_LDA_{df_name}/id2word.pkl'), 'rb') as f3:
        id2word = pickle.load(f3)
    
    df = pd.read_csv(os.path.join(df_path, 'body_clean.csv'))
    new_corpus = [Counter(text.split()) for text in df['Text']]
    word2id = {word: idx for idx, word in id2word.items()}
    
    formatted_corpus = [[(word2id[word], count) for word, count in text.items() if word in word2id] for text in new_corpus]
    lda_model = lda_results_all[0][0]
    
    corpus_lda = list(lda_model[formatted_corpus])
    date_list = df['Date'].tolist()
    
    topic_df = pd.DataFrame({'Date': date_list})
    for i in range(1, num_topics + 1):
        topic_df[f'Topic-{i}'] = 0
    
    for i, doc in enumerate(corpus_lda):
        for topic_id, score in doc:
            topic_df.at[i, f'Topic-{topic_id + 1}'] = score
    
    topic_df.to_csv(os.path.join(path_lda, f'{df_name}_topicDistributionDf.csv'), index=False)
    return topic_df


if __name__ == '__main__':
    path_lda = 'your path'
    df_path = 'your path'
    
    num_topics_dict = {'titles': 87, 'des': 100, 'body': 69}
    
    for name, num_topics in num_topics_dict.items():
        topics_features(name, path_lda, df_path, num_topics)
        print(f'{name} topic distribution computed.')
    
    for name in num_topics_dict.keys():
        df = pd.read_csv(os.path.join(path_lda, f'{name}_topicDistributionDf.csv'))
        df_grouped = df.groupby('Date').mean()
        df_grouped.to_csv(os.path.join(path_lda, f'{name}_topicDistributionDf_grouped.csv'))
        print(f'{name} topic distribution grouped and saved.')
