import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from utils import preprocess_data, build_tfidf_model

@st.cache_data
def calculate_enhanced_score(similarity_scores, ratings, prices, weights=None, _cache_key=None):
    """Calculate enhanced scoring with similarity, rating, and price factors"""
    if weights is None:
        weights = {'similarity': 0.6, 'rating': 0.3, 'price': 0.1}
    
    rating_scores = np.array(ratings) / 5.0
    scaler = MinMaxScaler()
    price_scores = 1 - scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()
    
    rating_scores = np.nan_to_num(rating_scores, nan=0.5)
    price_scores = np.nan_to_num(price_scores, nan=0.5)
    similarity_scores = np.nan_to_num(similarity_scores, nan=0.0)
    
    final_scores = (
        weights['similarity'] * similarity_scores +
        weights['rating'] * rating_scores +
        weights['price'] * price_scores
    )
    
    return final_scores

@st.cache_data
def enhanced_search(query, _df, _tfidf_model, _tfidf_matrix, top_k=10, _cache_key=None):
    """Enhanced search with TF-IDF similarity and multi-factor scoring"""
    processed_query = re.sub(r'[^a-zA-Z\s]', ' ', query.lower())
    processed_query = ' '.join(processed_query.split())
    
    query_vector = _tfidf_model.transform([processed_query])
    similarity_scores = cosine_similarity(query_vector, _tfidf_matrix).flatten()
    
    enhanced_scores = calculate_enhanced_score(
        similarity_scores, 
        _df['Rating'].values, 
        _df['Price'].values
    )
    
    result_df = _df.copy()
    result_df['similarity_score'] = similarity_scores
    result_df['final_score'] = enhanced_scores
    
    result_df = result_df.sort_values(['final_score', 'similarity_score'], ascending=[False, False])
    
    return result_df.head(top_k)

@st.cache_data
def initialize_app(_cache_key=None):
    """Initialize the application with data loading and preprocessing"""
    try:
        data = pd.read_csv("data/tourism_with_id.csv")
        
        required_columns = ['Place_Id', 'Place_Name', 'Description', 'Category', 
                          'City', 'Price', 'Rating', 'Lat', 'Long']
        
        for col in required_columns:
            if col not in data.columns:
                print(f"Missing required column: {col}")
                return None, None, None, None
        
        data = data[required_columns]
        
        data.dropna(subset=['Description', 'Lat'], inplace=True)
        data['Price'] = pd.to_numeric(data['Price'], errors='coerce').fillna(0)
        data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce').fillna(3.0)
        
        processed_data = preprocess_data(data)
        tfidf_model, tfidf_matrix = build_tfidf_model(processed_data)
        
        return data, processed_data, tfidf_model, tfidf_matrix
    except Exception as e:
        print(f"Error initializing app: {str(e)}")
        return None, None, None, None