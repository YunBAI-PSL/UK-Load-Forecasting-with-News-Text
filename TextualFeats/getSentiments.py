import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler


def compute_sentiment_scores(body_df, title_df, des_df):
    """Compute sentiment polarity and subjectivity for given text datasets."""
    polarity_title, polarity_des, polarity_body = [], [], []
    subject_title, subject_des, subject_body = [], [], []
    date_list = []
    
    for i in range(len(body_df)):
        title_text, des_text, body_text = title_df['Text'][i], des_df['Text'][i], body_df['Text'].iloc[i]
        
        if isinstance(title_text, str) and isinstance(des_text, str) and isinstance(body_text, str):
            tm_title, tm_des, tm_body = TextBlob(title_text), TextBlob(des_text), TextBlob(body_text)
            polarity_title.append(tm_title.sentiment.polarity)
            polarity_des.append(tm_des.sentiment.polarity)
            polarity_body.append(tm_body.sentiment.polarity)
                
            subject_title.append(tm_title.sentiment.subjectivity)
            subject_des.append(tm_des.sentiment.subjectivity)
            subject_body.append(tm_body.sentiment.subjectivity)
            date_list.append(body_df['Date'].iloc[i])
            
        if i % 100 == 0:
            print(f"Processing {i} / {len(body_df)}")
    
    new_df = pd.DataFrame({
        'Date': date_list,
        'polarity_title': polarity_title, 'polarity_des': polarity_des, 'polarity_body': polarity_body,
        'subject_title': subject_title, 'subject_des': subject_des, 'subject_body': subject_body
    })
    
    # Normalize sentiment scores
    scaler = MinMaxScaler()
    for col in new_df.columns[1:]:
        new_df[col] = scaler.fit_transform(new_df[col].values.reshape(-1, 1))
    
    return new_df


def compute_histogram_distribution(df):
    """Compute sentiment score distribution over different ranges."""
    hist_df = pd.DataFrame()
    max_grouped = df.groupby('Date').max()
    idx_list = max_grouped.index
    distribution = np.zeros((len(idx_list), 5))
    
    for j, idx in enumerate(idx_list):
        temp = df[df.iloc[:, 0] == idx]
        sub_distribution = np.zeros(5)
        temp_list = temp.iloc[:, 1].tolist()
        
        for val in temp_list:
            if val <= 0.2:
                sub_distribution[0] += 1
            elif val <= 0.4:
                sub_distribution[1] += 1
            elif val <= 0.6:
                sub_distribution[2] += 1
            elif val <= 0.8:
                sub_distribution[3] += 1
            else:
                sub_distribution[4] += 1
        
        distribution[j, :] = sub_distribution / len(temp)
    
    hist_df['Date'] = idx_list
    for i, rng in enumerate([(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]):
        hist_df[f'{df.columns[1]}_range{rng}'] = distribution[:, i]
    
    hist_df.set_index('Date', inplace=True)
    return hist_df


def compute_sentiment_statistics(df, column_name):    
    """Compute min, max, mean, std of sentiment scores grouped by date."""
    sub_df = df[['Date', column_name]]
    
    stats_df = pd.concat([
        sub_df.groupby('Date').agg(['max', 'min', 'mean', 'std']).rename(columns=lambda x: f'{column_name}_{x}'),
        compute_histogram_distribution(sub_df)
    ], axis=1)
    
    return stats_df


# Load datasets
df_path = ''
save_path = ''

title_df = pd.read_csv(os.path.join(df_path, 'titles_clean.csv'))
des_df = pd.read_csv(os.path.join(df_path, 'des_clean.csv'))
body_df = pd.read_csv(os.path.join(df_path, 'body_clean.csv'))

# Compute sentiment scores
sentiment_df = compute_sentiment_scores(body_df, title_df, des_df)

# Compute aggregated sentiment statistics
title_pol = compute_sentiment_statistics(sentiment_df, 'polarity_title')
title_sub = compute_sentiment_statistics(sentiment_df, 'subject_title')
des_pol = compute_sentiment_statistics(sentiment_df, 'polarity_des')
des_sub = compute_sentiment_statistics(sentiment_df, 'subject_des')
body_pol = compute_sentiment_statistics(sentiment_df, 'polarity_body')
body_sub = compute_sentiment_statistics(sentiment_df, 'subject_body')

# Save results
title_features = pd.concat([title_pol, title_sub], axis=1)
des_features = pd.concat([des_pol, des_sub], axis=1)
body_features = pd.concat([body_pol, body_sub], axis=1)

title_features.to_csv(os.path.join(save_path, 'title_features_senti.csv'))
des_features.to_csv(os.path.join(save_path, 'des_features_senti.csv'))
body_features.to_csv(os.path.join(save_path, 'body_features_senti.csv'))
