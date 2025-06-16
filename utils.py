import pandas as pd
import numpy as np
import requests
import re
from geopy.distance import geodesic
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

@st.cache_data
def load_data(_cache_key=None):
    """Load tourism data with validation"""
    try:
        df = pd.read_csv("data/tourism_with_id.csv")
        required_columns = ['Place_Id', 'Place_Name', 'Description', 'Category', 'City', 'Price', 'Rating', 'Lat', 'Long']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Kolom {col} tidak ditemukan dalam dataset!")
                return pd.DataFrame()
        df = df[required_columns]
        df.dropna(subset=['Description', 'Lat', 'Long'], inplace=True)
        return df
    except Exception as e:
        print(f"Error in loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def initialize_text_processors(_cache_key=None):
    """Initialize text processing tools with caching"""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        stop_factory = StemmerFactory()
        stemmer = stop_factory.create_stemmer()
        stopwords = set(['yang', 'dan', 'di', 'dari', 'ke', 'pada', 'untuk', 'dengan', 'adalah', 'ini', 'itu'])
        custom_stopwords = {'atau', 'akan', 'dapat', 'bisa', 'juga', 'sangat', 'lebih', 'serta', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        stopwords.update(custom_stopwords)
        return stemmer, stopwords, True
    except ImportError:
        stopwords = {'yang', 'ini', 'itu', 'dan', 'atau', 'dari', 'ke', 'di', 'pada', 'untuk', 'dengan', 'adalah', 'akan', 'dapat', 'bisa', 'juga', 'sangat', 'lebih', 'serta', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return None, stopwords, False

@st.cache_data
def preprocess_data(df, _cache_key=None):
    """Preprocess text data with optimized cleaning"""
    stemmer, stopwords, use_stemmer = initialize_text_processors()
    
    def optimized_clean_text(text):
        if pd.isna(text):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        if use_stemmer and stemmer:
            words = [stemmer.stem(word) for word in text.split() 
                    if len(word) > 2 and word not in stopwords]
        else:
            words = [word for word in text.split() 
                    if len(word) > 2 and word not in stopwords]
        return ' '.join(words)
    
    df = df.copy()
    with st.spinner("Memproses deskripsi teks..."):
        df['clean_desc'] = df['Description'].apply(optimized_clean_text)
    
    with st.spinner("Menggabungkan fitur teks..."):
        df['combined_text'] = (
            df['Place_Name'].str.lower().fillna('') + ' ' +
            df['Category'].str.lower().fillna('') + ' ' +
            df['City'].str.lower().fillna('') + ' ' +
            df['clean_desc'].fillna('')
        )
    
    return df

@st.cache_data
def build_tfidf_model(df, _cache_key=None):
    """Build TF-IDF model with optimized parameters"""
    with st.spinner("Membangun model TF-IDF..."):
        tfidf = TfidfVectorizer(
            max_features=2000,
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2),
            sublinear_tf=True,
            smooth_idf=False,
            use_idf=True,
            strip_accents='unicode',
            lowercase=True,
            stop_words=None,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    return tfidf, tfidf_matrix

@st.cache_data(ttl=3600)
def search_nearby_hotels(lat, lon, radius_km=5, _cache_key=None):
    """Search for hotels near a specific location using Overpass API"""
    try:
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["tourism"="hotel"](around:{radius_km*1000},{lat},{lon});
          node["tourism"="guest_house"](around:{radius_km*1000},{lat},{lon});
          node["tourism"="hostel"](around:{radius_km*1000},{lat},{lon});
          way["tourism"="hotel"](around:{radius_km*1000},{lat},{lon});
          way["tourism"="guest_house"](around:{radius_km*1000},{lat},{lon});
          way["tourism"="hostel"](around:{radius_km*1000},{lat},{lon});
        );
        out center meta;
        """
        
        response = requests.get(overpass_url, params={'data': query}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            hotels = []
            
            for element in data.get('elements', []):
                if 'lat' in element and 'lon' in element:
                    hotel_lat, hotel_lon = element['lat'], element['lon']
                elif 'center' in element:
                    hotel_lat, hotel_lon = element['center']['lat'], element['center']['lon']
                else:
                    continue
                
                tags = element.get('tags', {})
                name = tags.get('name', 'Hotel Tidak Diketahui')
                tourism_type = tags.get('tourism', 'hotel')
                distance = geodesic((lat, lon), (hotel_lat, hotel_lon)).kilometers
                
                hotels.append({
                    'name': name,
                    'type': tourism_type.replace('_', ' ').title(),
                    'Lat': hotel_lat,
                    'Long': hotel_lon,
                    'distance_km': round(distance, 2),
                    'address': tags.get('addr:full', tags.get('addr:street', 'Alamat tidak tersedia')),
                    'phone': tags.get('phone', 'Tidak tersedia'),
                    'website': tags.get('website', tags.get('contact:website', '')),
                    'amenities': tags.get('amenity', ''),
                    'stars': tags.get('stars', ''),
                    'email': tags.get('email', tags.get('contact:email', ''))
                })
            
            hotels = sorted(hotels, key=lambda x: x['distance_km'])
            return hotels[:50]
            
    except Exception as e:
        st.error(f"Error mencari hotel: {str(e)}")
        return []
    
    return []