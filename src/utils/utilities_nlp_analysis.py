# utilities_nlp_analysis.py
# ==============================
# Title: ADA Project NLP Analysis Utilities
# ==============================


import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
from sklearn.cluster import KMeans
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import os
import umap.umap_ as umap

# Function to translate text to English using batch translation
def translate_to_english_batch(texts):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate_batch(texts)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return texts

# Function to preprocess text
def preprocess_text(text, wtr):
    if isinstance(text, str):
        # Remove special characters and punctuation
        text = re.sub(r'\W+', ' ', text)
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        # Remove stop words and specific words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word not in wtr]
        # Join tokens back into a single string
        return ' '.join(tokens)
    else:
        return text


# Function to identify the top beer for each year in the United States
def identify_top_beers_us(ratings_df):
    review_counts = ratings_df.groupby(['beer_id']).size().reset_index(name='review_count')
    reviews_with_counts = pd.merge(ratings_df, review_counts, on=['beer_id'])
    filtered_reviews = reviews_with_counts[(reviews_with_counts['review_count'] >= 5) & (reviews_with_counts['location_user'] == 'United States')]

    weighted_avg_ratings = filtered_reviews.groupby(['beer_id', 'beer_name', 'year', 'review_count', 'location_user']).apply(
        lambda x: (x['rating'] * x['review_count']).sum() / x['review_count'].sum()
    ).reset_index(name='weighted_avg_rating')
    weighted_avg_ratings['rank'] = (
        weighted_avg_ratings.sort_values(['year', 'location_user', 'weighted_avg_rating', 'review_count'], 
                                         ascending=[True, True, False, False])
        .groupby(['year', 'location_user'])
        .cumcount() + 1
    )

    top_ranked_beers = weighted_avg_ratings[weighted_avg_ratings['rank'] == 1]
    return top_ranked_beers

# Function to get all reviews for the top beers from the United States, limited to 100 reviews per beer
def get_top_beer_reviews_us(top_ranked_beers, ratings_df):
    top_beer_reviews = pd.merge(top_ranked_beers, ratings_df, on='beer_id')
    top_beer_reviews_us = top_beer_reviews[top_beer_reviews['location_user_y'] == 'United States']
    
    # Limit to 100 reviews per beer
    top_beer_reviews_us = top_beer_reviews_us.groupby('beer_id').apply(lambda x: x.sample(min(len(x), 100), random_state=42)).reset_index(drop=True)
    
    return top_beer_reviews_us

# Function to preprocess and translate reviews
def preprocess_and_translate_reviews(reviews_df,wtr):  
    reviews_df['cleaned_text'] = reviews_df['text'].apply(lambda text: preprocess_text(text, wtr))
    return reviews_df

# Function to vectorize the texts
def vectorize_texts(reviews_df):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(reviews_df['cleaned_text'])
    return tfidf_matrix, vectorizer

# Function to perform UMAP
def perform_umap(tfidf_matrix):
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(tfidf_matrix.toarray())
    return umap_result

# Function to cluster the data
def cluster_data(umap_result):
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(umap_result)
    return clusters

# Function to assign years to clusters
def assign_years_to_clusters(reviews_df):
    year_cluster_mapping = {}
    for year in reviews_df['year_x'].unique():
        year_data = reviews_df[reviews_df['year_x'] == year]
        most_common_cluster = year_data['cluster'].value_counts().idxmax()
        year_cluster_mapping[year] = most_common_cluster
    return year_cluster_mapping

# Function to generate word clouds for each cluster or year
def generate_wordclouds(reviews_df, group_by='cluster'):
    grouped_data = reviews_df.groupby(group_by)
    for group, group_df in grouped_data:
        text = ' '.join(group_df['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for {group_by.capitalize()}: {group}")
        plt.show()

def generate_wordclouds_save(reviews_df, group_by='cluster'):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join('test', 'Sentiment', 'Worldclouds')
    os.makedirs(output_dir, exist_ok=True)

    grouped_data = reviews_df.groupby(group_by)
    for group, group_df in grouped_data:
        # Generate the word cloud text
        text = ' '.join(group_df['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Save the word cloud to the output directory
        file_path = os.path.join(output_dir, f'wordcloud_{group}.png')
        wordcloud.to_file(file_path)

        # Optionally, plot the word cloud for visual inspection
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for {group_by.capitalize()}: {group}")
        plt.show()

# Function to compute word frequencies by year
def compute_word_frequencies(reviews_df, year_column='year_x'):
    word_freq_by_year = {}
    for year, year_data in reviews_df.groupby(year_column):
        text = ' '.join(year_data['cleaned_text'])
        word_counts = Counter(text.split())
        word_freq_by_year[year] = word_counts
    return word_freq_by_year

# Improved Function to Plot Word Frequency Trends
def plot_word_frequency_trends(word_freq_by_year, words):
    # Prepare word trends
    
    word_trends = {word: [] for word in words}
    years = sorted(word_freq_by_year.keys())
    
    for year in years:
        year_counts = word_freq_by_year[year]
        for word in words:
            word_trends[word].append(year_counts.get(word, 0))
    
    # Create a more visually appealing plot
    plt.figure(figsize=(12, 8))
    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']  # Website-friendly color palette
    
    for i, (word, counts) in enumerate(word_trends.items()):
        plt.plot(
            years, counts, 
            label=word.capitalize(), 
            marker='o', markersize=8, 
            linewidth=2.5, 
            color=colors[i % len(colors)]
        )
    
    # Add labels, title, and customize aesthetics
    plt.xlabel('Year', fontsize=16, weight='bold', color='#333333')
    plt.ylabel('Frequency', fontsize=16, weight='bold', color='#333333')
    plt.title('Word Frequency Trends Over Years', fontsize=20, weight='bold', color='#333333')
    
    # Customize legend
    plt.legend(
        fontsize=12, 
        title="Words", 
        title_fontsize=14, 
        loc='upper left', 
        frameon=True, 
        framealpha=0.9, 
        edgecolor='#CCCCCC'
    )
    
    # Add grid for better readability
    plt.grid(color='#DDDDDD', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12, weight='bold', color='#555555')
    plt.yticks(fontsize=12, weight='bold', color='#555555')
    
    # Adjust layout for a clean appearance
    plt.tight_layout()
    
    # Optionally, save the plot for web use
    plt.savefig("test/Sentiment/word_frequency_trends_website.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# Function to compute sentiment for reviews
def compute_sentiment(reviews_df):
    reviews_df['sentiment'] = reviews_df['cleaned_text'].apply(
        lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0
    )
    sentiment_by_year = reviews_df.groupby('year_x')['sentiment'].mean()
    return sentiment_by_year

# Function to plot sentiment trends
def plot_sentiment_trends(sentiment_by_year):
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_by_year.index, sentiment_by_year.values, marker='o', label='Sentiment')
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment')
    plt.title('Sentiment Trends Over Years')
    plt.legend()
    plt.savefig("test/Sentiment/sentiment_trends.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# Function to extract cluster-specific keywords
def extract_cluster_keywords(reviews_df, cluster_column='cluster', n_top_words=10):
    cluster_texts = reviews_df.groupby(cluster_column)['cleaned_text'].apply(' '.join)
    cluster_keywords = {}
    for cluster, text in cluster_texts.items():
        word_counts = Counter(text.split())
        cluster_keywords[cluster] = word_counts.most_common(n_top_words)
    return cluster_keywords

# Function to visualize cluster profiles
def plot_cluster_profiles(cluster_keywords):
    for cluster, keywords in cluster_keywords.items():
        words, counts = zip(*keywords)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts, y=words)
        plt.title(f'Cluster {cluster} Profile')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.savefig(f"test/Sentiment/ClusterProfiles/clusterprofile_{cluster}.png", dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

def plot_cluster_profiles_save(cluster_keywords):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join('test', 'Sentiment', 'ClusterProfiles')
    os.makedirs(output_dir, exist_ok=True)

    for cluster, keywords in cluster_keywords.items():
        words, counts = zip(*keywords)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts, y=words)
        plt.title(f'Cluster {cluster} Profile')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        
        # Save the plot to the output directory
        file_path = os.path.join(output_dir, f'cluster_profile_{cluster}.png')
        plt.savefig(file_path, bbox_inches='tight')
        
        # Optionally, display the plot
        plt.show()

# Function to extract n-grams
def extract_ngrams(reviews_df, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    ngram_matrix = vectorizer.fit_transform(reviews_df['cleaned_text'])
    ngram_counts = ngram_matrix.sum(axis=0)
    ngrams = vectorizer.get_feature_names_out()
    ngram_freq = pd.DataFrame({'ngram': ngrams, 'frequency': ngram_counts.A1})
    ngram_freq = ngram_freq.sort_values(by='frequency', ascending=False)
    return ngram_freq

# Function to plot n-grams
def plot_top_ngrams(ngram_freq, top_n=20):
    top_ngrams = ngram_freq.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='frequency', y='ngram', data=top_ngrams)
    plt.title(f'Top {top_n} N-Grams')
    plt.xlabel('Frequency')
    plt.ylabel('N-Grams')
    plt.savefig("test/Sentiment/top_ngrams.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# Function to plot cluster distribution over years
def plot_cluster_distribution(reviews_df, year_column='year_x', cluster_column='cluster'):
    cluster_distribution = reviews_df.groupby([year_column, cluster_column]).size().unstack(fill_value=0)
    
    cluster_distribution.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Cluster Distribution Over Years')
    plt.xlabel('Year')
    plt.ylabel('Review Count')
    plt.legend(title='Cluster')
    plt.savefig("test/Sentiment/cluster_distribution.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()



# Function to compute the most used words for each cluster
def compute_most_used_words(reviews_df, cluster_col='cluster', text_col='cleaned_text'):
    cluster_word_counts = {}
    for cluster in reviews_df[cluster_col].unique():
        cluster_data = reviews_df[reviews_df[cluster_col] == cluster]
        all_tokens = [token for tokens in cluster_data[text_col].apply(word_tokenize) for token in tokens]
        token_counts = Counter(all_tokens)
        most_common_words = token_counts.most_common(20)
        cluster_word_counts[cluster] = most_common_words
    return cluster_word_counts
