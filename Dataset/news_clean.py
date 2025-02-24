#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for text cleaning and preprocessing.

Created on: Jun 27, 2022
Updated for GitHub release.
"""

import warnings
import pandas as pd
import nltk
from multiprocessing import Pool
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Suppress warnings for clarity
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize tokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')

def get_train(path):
    """Load dataset and filter necessary columns."""
    if path.endswith('.xlsx'):
        raw_df = pd.read_excel(path)
    else:
        raw_df = pd.read_csv(path)
    
    raw_df = raw_df[['title', 'published_date', 'description', 'content']].dropna()
    raw_df = raw_df[(raw_df['published_date'] >= '2016-06-01') & (raw_df['published_date'] <= '2021-05-31')]
    
    return raw_df[['published_date', 'title']], raw_df[['published_date', 'description']], raw_df[['published_date', 'content']].dropna()

def text_preprocessing(i, corpus):
    """Perform text cleaning and stopword removal."""
    date = corpus.iloc[:, 0].tolist()
    text = corpus.iloc[:, 1].tolist()
    
    entry = tokenizer.tokenize(text[i].lower())
    final_words = ' '.join([word for word in entry if word not in stopwords.words('english')])
    
    return {'Date': date[i], 'Text': final_words}

def work(z):
    return text_preprocessing(z[0], z[1])

def multi_preprocess(text_type):
    """Parallel processing of text data."""
    data_list = [[i, text_type] for i in range(len(text_type))]
    
    with Pool(7) as p:
        clean_dicts = p.map(work, data_list)
    
    return pd.DataFrame(clean_dicts)

if __name__ == '__main__':
    # Set input and output file paths
    input_path = "data/EMnews.csv"
    output_path = "data/EMnews_clean.csv"
    
    # Process data
    _, _, body = get_train(input_path)
    body_clean = multi_preprocess(body)
    body_clean.to_csv(output_path, index=False)
    print(f"Cleaned text data saved to {output_path}")
