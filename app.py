import streamlit as st
from streamlit_folium import st_folium
import folium
from search import initialize_app, enhanced_search
from utils import search_nearby_hotels
from astar import calculate_optimized_astar_path
import pandas as pd
import time

# Set page config with custom styling
st.set_page_config(
    page_title="Urbanmories - Travel Recommendation System",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS using Streamlit's approach
def apply_custom_styling():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global font */
    .stApp {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        border-radius: 20px;
        border: none;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Custom sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    defaults = {
        'page': 'search',
        'selected_destination': None,
        'hotels_data': None,
        'hotel_radius': 5,
        'show_hotels': False,
        'selected_hotel_path': None,
        'app_fully_initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Enhanced preload with better progress visualization
def preload_all_data():
    if st.session_state.app_fully_initialized:
        return True
    
    # Create loading interface
    st.markdown("## 🚀 Memuat Urbanmories")
    st.markdown("*Sedang menginisialisasi sistem rekomendasi cerdas...*")
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        steps = [
            ("📊 Memuat dataset wisata...", 25),
            ("🤖 Menginisialisasi model TF-IDF...", 50),
            ("🔍 Memproses matrix similarity...", 75),
            ("✅ Aplikasi siap digunakan!", 100)
        ]
        
        for i, (status_text, progress) in enumerate(steps):
            status_placeholder.info(status_text)
            progress_placeholder.progress(progress)
            
            if i == 1:  # Initialize on second step
                data, processed_data, tfidf_model, tfidf_matrix = initialize_app()
                st.session_state.data = data
                st.session_state.processed_data = processed_data
                st.session_state.tfidf_model = tfidf_model
                st.session_state.tfidf_matrix = tfidf_matrix
            
            time.sleep(0.5)
        
        st.session_state.app_fully_initialized = True
        
        # Clear loading elements
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Show success message
        st.success("🎉 Urbanmories berhasil dimuat! Selamat menjelajahi destinasi wisata terbaik Indonesia!")
        time.sleep(1)
        st.rerun()
        
        return True
    except Exception as e:
        st.error(f"❌ Error saat inisialisasi aplikasi: {str(e)}")
        return False

# Enhanced search page with pure Streamlit
def show_search_page():
    # Main header with enhanced styling
    st.markdown("# 🏖️ URBANMORIES")
    st.markdown("### *Temukan destinasi wisata terbaik di Indonesia dengan sistem pencarian cerdas*")
    st.markdown("---")

    # Enhanced sidebar with better organization
    with st.sidebar:
        st.header("⚙️ Pengaturan Pencarian")
        
        with st.expander("🎛️ Bobot Pencarian", expanded=True):
            st.caption("Sesuaikan prioritas pencarian Anda")
            search_weights = {
                'similarity': st.slider("🎯 Relevansi Teks", 0.1, 1.0, 0.6, 0.1, 
                                       help="Seberapa penting kesesuaian kata kunci"),
                'rating': st.slider("⭐ Rating Destinasi", 0.1, 1.0, 0.3, 0.1, 
                                   help="Seberapa penting rating pengguna"),
                'price': st.slider("💰 Faktor Harga", 0.1, 1.0, 0.1, 0.1, 
                                  help="Seberapa penting harga terjangkau")
            }
        
        # Quick stats with metrics
        if 'processed_data' in st.session_state:
            st.header("📊 Statistik Database")
            
            data = st.session_state.processed_data
            total_destinations = len(data)
            total_cities = data['City'].nunique()
            total_categories = data['Category'].nunique()
            avg_rating = data['Rating'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏖️ Destinasi", total_destinations)
                st.metric("🏙️ Kota", total_cities)
            with col2:
                st.metric("🏷️ Kategori", total_categories)
                st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")

    # Search interface with enhanced layout
    st.header("🔍 Pencarian Destinasi")
    
    # Search input with better styling
    query = st.text_input(
        "Masukkan kata kunci pencarian:",
        placeholder="Contoh: 'pantai indah untuk keluarga', 'gunung hiking adventure'",
        help="Gunakan kata kunci yang spesifik untuk hasil terbaik"
    )

    # Enhanced filter section
    st.subheader("🎯 Filter Pencarian")
    
    # Create columns for filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        kota_options = ["Semua Kota"] + sorted(st.session_state.processed_data['City'].unique())
        kota = st.selectbox("🏙️ Pilih Kota", kota_options)
    
    with col2:
        kategori_options = ["Semua Kategori"] + sorted(st.session_state.processed_data['Category'].unique())
        kategori = st.selectbox("🏷️ Kategori Wisata", kategori_options)
    
    with col3:
        min_rating = st.slider("⭐ Rating Minimal", 0.0, 5.0, 0.0, 0.1)
    
    with col4:
        max_harga = st.number_input("💰 Budget Max (Rp)", value=1000000, step=50000)

    # Advanced options
    with st.expander("🔧 Pengaturan Lanjutan"):
        col5, col6 = st.columns(2)
        with col5:
            show_scores = st.checkbox("📊 Tampilkan skor detail")
            min_similarity = st.slider("🎯 Threshold relevansi", 0.0, 1.0, 0.1, 0.05)
        with col6:
            result_count = st.slider("📋 Maksimal hasil", 5, 20, 10)

    # Execute search
    if query:
        with st.spinner("🔍 Mencari destinasi terbaik untuk Anda..."):
            results = enhanced_search(
                query, 
                st.session_state.processed_data, 
                st.session_state.tfidf_model, 
                st.session_state.tfidf_matrix, 
                top_k=result_count
            )
            
            # Apply filters
            if kota != "Semua Kota":
                results = results[results['City'] == kota]
            if kategori != "Semua Kategori":
                results = results[results['Category'] == kategori]
            
            results = results[
                (results['Rating'] >= min_rating) & 
                (results['Price'] <= max_harga) &
                (results['similarity_score'] >= min_similarity)
            ]
            
            # Calculate final scores
            if not results.empty:
                results['final_score'] = results.apply(
                    lambda row: (
                        search_weights['similarity'] * row['similarity_score'] +
                        search_weights['rating'] * (row['Rating'] / 5.0) +
                        search_weights['price'] * (1 - (row['Price'] - st.session_state.processed_data['Price'].min()) / 
                        (st.session_state.processed_data['Price'].max() - st.session_state.processed_data['Price'].min() + 1e-6))
                    ), axis=1
                )
                results = results.sort_values('final_score', ascending=False)

        # Display results
        if results.empty:
            st.warning("⚠️ Tidak ada destinasi yang cocok dengan kriteria Anda. Coba ubah filter atau kata kunci.")
        else:
            st.success(f"🎉 Ditemukan {len(results)} destinasi terbaik!")
            st.markdown("---")
            
            # Display results with enhanced cards
            for idx, (i, row) in enumerate(results.iterrows(), 1):
                with st.container():
                    # Ranking and basic info
                    col_rank, col_info, col_metrics = st.columns([0.5, 2.5, 1])
                    
                    with col_rank:
                        # Ranking with emoji
                        rank_emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}️⃣"
                        st.markdown(f"## {rank_emoji}")
                    
                    with col_info:
                        st.markdown(f"### {row['Place_Name']}")
                        st.markdown(f"**📍 {row['City']} • 🏷️ {row['Category']}**")
                        
                        # Price and rating
                        col_price, col_rating = st.columns(2)
                        with col_price:
                            st.markdown(f"**💰 Rp {row['Price']:,.0f}**")
                        with col_rating:
                            star_rating = "⭐" * int(row['Rating']) + "☆" * (5 - int(row['Rating']))
                            st.markdown(f"**{star_rating} {row['Rating']}/5**")
                        
                        # Description
                        description = row['Description'][:150] + "..." if len(row['Description']) > 150 else row['Description']
                        st.caption(description)
                    
                    with col_metrics:
                        if show_scores:
                            st.metric("🎯 Relevansi", f"{row['similarity_score']:.3f}")
                            st.metric("🏆 Skor Final", f"{row['final_score']:.3f}")
                    
                    # Action button
                    col_spacer, col_button = st.columns([3, 1])
                    with col_button:
                        if st.button("📋 Lihat Detail", key=f"detail_{row['Place_Id']}", type="primary"):
                            st.session_state.selected_destination = row
                            st.session_state.page = 'detail'
                            st.session_state.hotels_data = None
                            st.session_state.show_hotels = False
                            st.session_state.selected_hotel_path = None
                            st.rerun()
                    
                    st.markdown("---")

# Enhanced detail page
def show_detail_page():
    if st.session_state.selected_destination is None:
        st.error("Tidak ada destinasi yang dipilih!")
        if st.button("🔙 Kembali ke Pencarian"):
            st.session_state.page = 'search'
            st.rerun()
        return
    
    dest = st.session_state.selected_destination
    
    # Header with back button
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("🔙 Kembali", type="secondary"):
            st.session_state.page = 'search'
            st.session_state.hotels_data = None
            st.session_state.show_hotels = False
            st.session_state.selected_hotel_path = None
            st.rerun()
    
    with col_title:
        st.markdown(f"# 🏖️ {dest['Place_Name']}")
        st.markdown(f"**📍 {dest['City']} • 🏷️ {dest['Category']} • ⭐ {dest['Rating']}/5**")
    
    st.markdown("---")
    
    # Destination details
    col_detail, col_map = st.columns([2, 1])
    
    with col_detail:
        st.subheader("📋 Informasi Lengkap Destinasi")
        
        # Information in columns
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"""
            **📍 Lokasi:** {dest['City']}  
            **🏷️ Kategori:** {dest['Category']}  
            **🌐 Koordinat:** {dest['Lat']:.6f}, {dest['Long']:.6f}
            """)
        with info_col2:
            st.markdown(f"""
            **💰 Harga:** Rp {dest['Price']:,.0f}  
            **⭐ Rating:** {dest['Rating']}/5  
            **🆔 ID:** {dest['Place_Id']}
            """)
        
        st.subheader("📖 Deskripsi")
        st.write(dest['Description'])
    
    with col_map:
        st.subheader("🗺️ Lokasi di Peta")
        
        # Create map
        dest_map = folium.Map(
            location=[dest['Lat'], dest['Long']], 
            zoom_start=15
        )
        
        folium.Marker(
            [dest['Lat'], dest['Long']],
            popup=f"{dest['Place_Name']}<br>{dest['City']}<br>⭐ {dest['Rating']}/5",
            tooltip=dest['Place_Name'],
            icon=folium.Icon(color='red', icon='star')
        ).add_to(dest_map)
        
        st_folium(dest_map, width=400, height=300)
    
    # Hotel search section
    st.markdown("---")
    st.header("🏨 Pencarian Hotel Terdekat")
    
    col_search1, col_search2, col_search3 = st.columns(3)
    
    with col_search1:
        search_radius = st.slider("🔍 Radius (km)", 1, 15, st.session_state.hotel_radius)
        st.session_state.hotel_radius = search_radius
    
    with col_search2:
        if st.button("🔍 Cari Hotel", type="primary"):
            with st.spinner("🏨 Mencari hotel terdekat..."):
                hotels = search_nearby_hotels(dest['Lat'], dest['Long'], search_radius)
                st.session_state.hotels_data = hotels
                st.session_state.show_hotels = True
                st.session_state.selected_hotel_path = None
                st.rerun()
    
    with col_search3:
        if st.session_state.hotels_data and st.button("🗺️ Tampilkan Peta"):
            st.session_state.show_hotels = True

    # Display hotels
    if st.session_state.show_hotels and st.session_state.hotels_data:
        hotels = st.session_state.hotels_data
        
        if hotels:
            st.success(f"🎉 Ditemukan {len(hotels)} hotel dalam radius {search_radius} km")
            
            # Hotel list
            for idx, hotel in enumerate(hotels):
                with st.container():
                    col_hotel_info, col_hotel_metrics, col_hotel_action = st.columns([2.5, 1, 1])
                    
                    with col_hotel_info:
                        st.markdown(f"### {hotel['name']}")
                        st.markdown(f"**🏨 {hotel['type']}**")
                        
                        if hotel['address'] != 'Alamat tidak tersedia':
                            address_short = hotel['address'][:60] + "..." if len(hotel['address']) > 60 else hotel['address']
                            st.caption(f"🏠 {address_short}")
                        
                        if hotel['phone'] != 'Tidak tersedia':
                            st.caption(f"📞 {hotel['phone']}")
                        
                        if hotel['website']:
                            st.markdown(f"🌐 [Website]({hotel['website']})")
                    
                    with col_hotel_metrics:
                        # Distance with color coding
                        distance = hotel['distance_km']
                        if distance <= 2:
                            st.success(f"📍 {distance} km")
                        elif distance <= 5:
                            st.warning(f"📍 {distance} km")
                        else:
                            st.info(f"📍 {distance} km")
                    
                    with col_hotel_action:
                        if st.button("🗺️ Lihat Rute", key=f"route_{idx}"):
                            with st.spinner("🧭 Menghitung rute..."):
                                origin = (dest['Lat'], dest['Long'])
                                destination = (hotel['Lat'], hotel['Long'])
                                
                                path_result = calculate_optimized_astar_path(origin, destination)
                                
                                st.session_state.selected_hotel_path = {
                                    'hotel': hotel,
                                    'path_data': path_result
                                }
                                st.rerun()
                    
                    st.markdown("---")
            
            # Interactive map
            st.subheader("🗺️ Peta Interaktif Hotel & Rute")
            
            # Create map
            hotel_map = folium.Map(
                location=[dest['Lat'], dest['Long']],
                zoom_start=12
            )
            
            # Add destination marker
            folium.Marker(
                [dest['Lat'], dest['Long']],
                popup=f"🏖️ {dest['Place_Name']}<br>📍 Destinasi Utama<br>⭐ {dest['Rating']}/5",
                tooltip=f"🏖️ {dest['Place_Name']}",
                icon=folium.Icon(color='red', icon='star')
            ).add_to(hotel_map)
            
            # Add hotel markers
            for idx, hotel in enumerate(hotels):
                color = 'green' if hotel['distance_km'] <= 2 else 'orange' if hotel['distance_km'] <= 5 else 'blue'
                
                folium.Marker(
                    [hotel['Lat'], hotel['Long']],
                    popup=f"🏨 {hotel['name']}<br>📍 {hotel['distance_km']} km<br>🏨 {hotel['type']}",
                    tooltip=f"🏨 {hotel['name']}",
                    icon=folium.Icon(color=color, icon='bed')
                ).add_to(hotel_map)
            
            # Add route if selected
            if st.session_state.selected_hotel_path:
                path_data = st.session_state.selected_hotel_path['path_data']
                selected_hotel = st.session_state.selected_hotel_path['hotel']
                
                if 'path' in path_data and len(path_data['path']) > 1:
                    folium.PolyLine(
                        locations=path_data['path'],
                        weight=5,
                        color='red',
                        opacity=0.8,
                        popup=f"🛣️ Rute ke {selected_hotel['name']}"
                    ).add_to(hotel_map)
                
                # Route information
                st.info(f"""
                **🗺️ Rute ke {selected_hotel['name']}**
                
                📏 **Jarak:** {path_data['distance']:.2f} km  
                🔍 **Metode:** {path_data.get('method', 'A* Road-based')}  
                🔄 **Iterasi:** {path_data.get('iterations', 'N/A')}  
                """ + (f"⏱️ **Estimasi Waktu:** {path_data['duration_minutes']:.0f} menit" if 'duration_minutes' in path_data else ""))
                
                # Navigation steps
                if 'steps' in path_data and path_data['steps']:
                    with st.expander("🧭 Panduan Navigasi Detail"):
                        for i, step in enumerate(path_data['steps'][:10], 1):
                            icon = "🚗" if i == 1 else "➡️" if i < 10 else "🏁"
                            st.markdown(f"**{icon} Langkah {i}:** {step.get('instruction', 'Continue')}")
                            st.caption(f"📍 {step.get('name', 'Unnamed road')} • 📏 {step.get('distance', 0):.0f}m")
            
            # Display the map
            st_folium(hotel_map, width=800, height=500)
            
        else:
            st.warning(f"❌ Tidak ditemukan hotel dalam radius {search_radius} km. Coba perluas radius pencarian.")

# Main function
def main():
    # Apply custom styling
    apply_custom_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Preload data
    if not preload_all_data():
        st.stop()
    
    # Navigation
    if st.session_state.page == 'search':
        show_search_page()
    elif st.session_state.page == 'detail':
        show_detail_page()
    
    # Enhanced footer using native Streamlit
    st.markdown("---")
    st.markdown("## 🏖️ URBANMORIES")
    st.markdown("**Sistem Rekomendasi Wisata Cerdas Indonesia**")
    
    # Footer information in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 Teknologi**
        - TF-IDF Vector Space Model
        - Cosine Similarity Matching
        - A* Road-Following Algorithm
        - Multi-criteria Decision Making
        """)
    
    with col2:
        st.markdown("""
        **🗺️ Sumber Data**
        - Database Wisata Indonesia
        - OpenStreetMap (OSM)
        - Open Source Routing Machine (OSRM)
        - Real-time Hotel Data
        """)
    
    with col3:
        st.markdown("""
        **✨ Fitur Unggulan**
        - Pencarian Berbasis AI
        - Filter Multi-kriteria
        - Visualisasi Peta Interaktif
        - Optimasi Rute Real-time
        """)
    
    st.markdown("---")
    st.markdown("🚀 Made with ❤️ for Indonesian Tourism")

# Run the app
if __name__ == "__main__":
    main()