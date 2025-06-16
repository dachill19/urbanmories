import streamlit as st
from streamlit_folium import st_folium
import folium
from search import initialize_app, enhanced_search
from utils import search_nearby_hotels
from astar import calculate_optimized_astar_path
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Urban Explorer",
    page_icon="🏖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'search'
    if 'selected_destination' not in st.session_state:
        st.session_state.selected_destination = None
    if 'hotels_data' not in st.session_state:
        st.session_state.hotels_data = None
    if 'hotel_radius' not in st.session_state:
        st.session_state.hotel_radius = 5
    if 'show_hotels' not in st.session_state:
        st.session_state.show_hotels = False
    if 'selected_hotel_path' not in st.session_state:
        st.session_state.selected_hotel_path = None
    if 'app_fully_initialized' not in st.session_state:
        st.session_state.app_fully_initialized = False

# Preload all data
def preload_all_data():
    if st.session_state.app_fully_initialized:
        return True
    
    with st.spinner("🚀 Menginisialisasi Urban Explorer - Memuat semua data..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📊 Memuat dataset dan model...")
            data, processed_data, tfidf_model, tfidf_matrix = initialize_app()
            st.session_state.data = data
            st.session_state.processed_data = processed_data
            st.session_state.tfidf_model = tfidf_model
            st.session_state.tfidf_matrix = tfidf_matrix
            progress_bar.progress(100)
            status_text.text("✅ Semua data berhasil dimuat! Aplikasi siap digunakan.")
            
            st.session_state.app_fully_initialized = True
            
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return True
        except Exception as e:
            st.error(f"❌ Error saat inisialisasi aplikasi: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            return False

# Search page
def show_search_page():
    st.title("🏖 URBAN EXPLORER")
    st.markdown("Temukan destinasi wisata terbaik di Indonesia dengan sistem pencarian cerdas yang menggabungkan relevansi, rating, dan harga.")

    # Sidebar configuration
    st.sidebar.header("⚙ Pengaturan Pencarian")
    search_weights = {
        'similarity': st.sidebar.slider("Bobot Relevansi Teks", 0.1, 1.0, 0.6, 0.1),
        'rating': st.sidebar.slider("Bobot Rating", 0.1, 1.0, 0.3, 0.1),
        'price': st.sidebar.slider("Bobot Harga", 0.1, 1.0, 0.1, 0.1)
    }

    # Search interface
    query = st.text_input("🔍 Cari destinasi wisata", "", placeholder="Contoh: 'pantai indah untuk keluarga', 'gunung hiking adventure'")

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kota = st.selectbox("🏙 Kota", ["Semua Kota"] + sorted(st.session_state.processed_data['City'].unique()))
    with col2:
        kategori = st.selectbox("🏷 Kategori", ["Semua Kategori"] + sorted(st.session_state.processed_data['Category'].unique()))
    with col3:
        min_rating = st.slider("⭐ Rating minimal", 0.0, 5.0, 0.0, 0.1)
    with col4:
        max_harga = st.number_input("💰 Harga maksimal (Rp)", value=1000000, step=50000)

    # Advanced options
    with st.expander("🔧 Opsi Pencarian Lanjutan"):
        col5, col6 = st.columns(2)
        with col5:
            show_similarity_scores = st.checkbox("Tampilkan skor similaritas", False)
            min_similarity = st.slider("Threshold similaritas minimum", 0.0, 1.0, 0.1, 0.05)
        with col6:
            result_count = st.slider("Jumlah hasil maksimal", 5, 20, 10)

    # Execute search
    if query:
        with st.spinner("🔍 Mencari destinasi terbaik..."):
            results = enhanced_search(query, st.session_state.processed_data, st.session_state.tfidf_model, st.session_state.tfidf_matrix, top_k=result_count)
            
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
            
            if not results.empty:
                results['final_score'] = st.session_state.processed_data.loc[results.index, 'final_score'] = results.apply(
                    lambda row: (
                        search_weights['similarity'] * row['similarity_score'] +
                        search_weights['rating'] * (row['Rating'] / 5.0) +
                        search_weights['price'] * (1 - (row['Price'] - st.session_state.processed_data['Price'].min()) / 
                        (st.session_state.processed_data['Price'].max() - st.session_state.processed_data['Price'].min() + 1e-6))
                    ), axis=1
                )
                results = results.sort_values('final_score', ascending=False)

        if results.empty:
            st.warning("⚠ Tidak ada destinasi yang cocok dengan kriteria pencarian Anda.")
        else:
            st.success(f"✅ Ditemukan {len(results)} destinasi yang cocok!")
            st.subheader(f"🎯 Top {len(results)} Rekomendasi Destinasi")
            
            for idx, (i, row) in enumerate(results.iterrows(), 1):
                with st.container():
                    col_info, col_metrics, col_action = st.columns([3, 1, 1])
                    
                    with col_info:
                        st.markdown(f"### {idx}. {row['Place_Name']}")
                        st.markdown(f"📍 *{row['City']}* • 🏷 {row['Category']}")
                        st.markdown(f"💰 *Rp {row['Price']:,.0f}* • ⭐ *{row['Rating']}/5*")
                        
                        description = row['Description']
                        if len(description) > 200:
                            description = description[:200] + "..."
                        st.markdown(f"📝 {description}")
                    
                    with col_metrics:
                        if show_similarity_scores:
                            st.metric("🎯 Relevansi", f"{row['similarity_score']:.3f}")
                            st.metric("🏆 Skor Akhir", f"{row['final_score']:.3f}")
                    
                    with col_action:
                        if st.button(f"📋 Detail", key=f"detail_{row['Place_Id']}", type="primary"):
                            st.session_state.selected_destination = row
                            st.session_state.page = 'detail'
                            st.session_state.hotels_data = None
                            st.session_state.show_hotels = False
                            st.session_state.selected_hotel_path = None
                            st.rerun()
                    
                    st.markdown("---")

# Detail page
def show_detail_page():
    if st.session_state.selected_destination is None:
        st.error("Tidak ada destinasi yang dipilih!")
        if st.button("🔙 Kembali ke Pencarian"):
            st.session_state.page = 'search'
            st.rerun()
        return
    
    dest = st.session_state.selected_destination
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"🏖 {dest['Place_Name']}")
    with col2:
        if st.button("🔙 Kembali ke Pencarian", type="secondary"):
            st.session_state.page = 'search'
            st.session_state.hotels_data = None
            st.session_state.show_hotels = False
            st.session_state.selected_hotel_path = None
            st.rerun()
    
    # Destination details
    st.markdown("---")
    
    col_detail1, col_detail2 = st.columns([2, 1])
    
    with col_detail1:
        st.subheader("📋 Informasi Destinasi")
        st.markdown(f"📍 *Lokasi:* {dest['City']}")
        st.markdown(f"🏷 *Kategori:* {dest['Category']}")
        st.markdown(f"💰 *Harga:* Rp {dest['Price']:,.0f}")
        st.markdown(f"⭐ *Rating:* {dest['Rating']}/5")
        st.markdown(f"🌐 *Koordinat:* {dest['Lat']:.6f}, {dest['Long']:.6f}")
        
        st.subheader("📖 Deskripsi")
        st.markdown(dest['Description'])
    
    with col_detail2:
        st.subheader("🗺 Lokasi di Peta")
        dest_map = folium.Map(
            location=[dest['Lat'], dest['Long']], 
            zoom_start=15,
            tiles='OpenStreetMap'
        )
        
        folium.Marker(
            [dest['Lat'], dest['Long']],
            popup=f"<b>{dest['Place_Name']}</b><br>{dest['City']}<br>⭐ {dest['Rating']}/5",
            tooltip=dest['Place_Name'],
            icon=folium.Icon(color='red', icon='star')
        ).add_to(dest_map)
        
        st_folium(dest_map, width=400, height=300)
    
    # Hotel search section
    st.markdown("---")
    st.subheader("🏨 Pencarian Hotel Terdekat")
    
    col_hotel1, col_hotel2, col_hotel3 = st.columns([1, 1, 1])
    
    with col_hotel1:
        search_radius = st.slider("🔍 Radius pencarian (km)", 1, 15, st.session_state.hotel_radius, 1)
        st.session_state.hotel_radius = search_radius
    
    with col_hotel2:
        if st.button("🔍 Cari Hotel", type="primary"):
            with st.spinner("Mencari hotel terdekat..."):
                hotels = search_nearby_hotels(dest['Lat'], dest['Long'], search_radius)
                st.session_state.hotels_data = hotels
                st.session_state.show_hotels = True
                st.session_state.selected_hotel_path = None
                st.rerun()
    
    with col_hotel3:
        if st.session_state.hotels_data and st.button("🗺 Tampilkan di Peta"):
            st.session_state.show_hotels = True
    
    # Display hotels if available
    if st.session_state.show_hotels and st.session_state.hotels_data:
        hotels = st.session_state.hotels_data
        
        if hotels:
            st.success(f"✅ Ditemukan {len(hotels)} hotel dalam radius {search_radius} km")
            
            st.subheader("📋 Daftar Hotel")
            
            for idx, hotel in enumerate(hotels):
                with st.container():
                    col_hotel_info, col_hotel_action = st.columns([3, 1])
                    
                    with col_hotel_info:
                        st.markdown(f"**{hotel['name']}** ({hotel['type']})")
                        st.markdown(f"📍 {hotel['distance_km']} km dari destinasi")
                        if hotel['address'] != 'Alamat tidak tersedia':
                            st.markdown(f"🏠 {hotel['address']}")
                        if hotel['phone'] != 'Tidak tersedia':
                            st.markdown(f"📞 {hotel['phone']}")
                        if hotel['website']:
                            st.markdown(f"🌐 [Website]({hotel['website']})")
                    
                    with col_hotel_action:
                        if st.button(f"🗺 Lihat Rute", key=f"route_{idx}"):
                            with st.spinner("Menghitung rute optimal..."):
                                origin = (dest['Lat'], dest['Long'])
                                destination = (hotel['Lat'], hotel['Long'])
                                
                                path_result = calculate_optimized_astar_path(origin, destination)
                                
                                st.session_state.selected_hotel_path = {
                                    'hotel': hotel,
                                    'path_data': path_result
                                }
                                st.rerun()
            
            # Map with hotels and routes
            st.subheader("🗺 Peta Hotel dan Rute")
            
            center_lat = dest['Lat']
            center_lon = dest['Long']
            
            hotel_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            folium.Marker(
                [dest['Lat'], dest['Long']],
                popup=f"<b>{dest['Place_Name']}</b><br>📍 Destinasi Utama<br>⭐ {dest['Rating']}/5",
                tooltip=dest['Place_Name'],
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(hotel_map)
            
            for idx, hotel in enumerate(hotels):
                if hotel['distance_km'] <= 2:
                    color = 'green'
                elif hotel['distance_km'] <= 5:
                    color = 'orange'
                else:
                    color = 'blue'
                
                popup_text = f"""
                <b>{hotel['name']}</b><br>
                {hotel['type']}<br>
                📍 {hotel['distance_km']} km<br>
                🏠 {hotel['address'][:50]}{'...' if len(hotel['address']) > 50 else ''}
                """
                
                folium.Marker(
                    [hotel['Lat'], hotel['Long']],
                    popup=popup_text,
                    tooltip=hotel['name'],
                    icon=folium.Icon(color=color, icon='bed', prefix='fa')
                ).add_to(hotel_map)
            
            if st.session_state.selected_hotel_path:
                path_data = st.session_state.selected_hotel_path['path_data']
                selected_hotel = st.session_state.selected_hotel_path['hotel']
                
                if 'path' in path_data and len(path_data['path']) > 1:
                    folium.PolyLine(
                        locations=path_data['path'],
                        weight=5,
                        color='red',
                        opacity=0.8,
                        popup=f"Rute ke {selected_hotel['name']}"
                    ).add_to(hotel_map)
                    
                    if 'visited_nodes' in path_data and len(path_data['visited_nodes']) > 1:
                        folium.PolyLine(
                            locations=path_data['visited_nodes'],
                            weight=2,
                            color='blue',
                            opacity=0.5,
                            dash_array='5, 5',
                            popup="Node yang dikunjungi A*"
                        ).add_to(hotel_map)
                
                st.info(f"""
                🗺 *Rute ke {selected_hotel['name']}*
                - 📏 Jarak: {path_data['distance']:.2f} km
                - 🔍 Metode: {path_data.get('method', 'A* Road-based')}
                - 🔄 Iterasi: {path_data.get('iterations', 'N/A')}
                - ℹ {path_data.get('note', 'Rute dihitung menggunakan algoritma A*')}
                """)
                
                if 'duration_minutes' in path_data:
                    st.info(f"⏱ *Estimasi Waktu Tempuh:* {path_data['duration_minutes']:.0f} menit")
                
                if 'steps' in path_data and path_data['steps']:
                    with st.expander("🧭 Panduan Navigasi"):
                        for i, step in enumerate(path_data['steps'][:10], 1):
                            st.markdown(f"{i}. *{step.get('instruction', 'Continue')}* - {step.get('name', 'Unnamed road')} ({step.get('distance', 0):.0f}m)")
            
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 150px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:12px; padding: 10px">
            <p><b>Legenda:</b></p>
            <p><i class="fa fa-star" style="color:red"></i> Destinasi</p>
            <p><i class="fa fa-bed" style="color:green"></i> Hotel ≤2km</p>
            <p><i class="fa fa-bed" style="color:orange"></i> Hotel ≤5km</p>
            <p><i class="fa fa-bed" style="color:blue"></i> Hotel >5km</p>
            </div>
            '''
            hotel_map.get_root().html.add_child(folium.Element(legend_html))
            
            st_folium(hotel_map, width=800, height=500)
            
        else:
            st.warning(f"❌ Tidak ditemukan hotel dalam radius {search_radius} km. Coba perluas radius pencarian.")

# Main function
def main():
    initialize_session_state()
    
    if not preload_all_data():
        st.stop()
    
    if st.session_state.page == 'search':
        show_search_page()
    elif st.session_state.page == 'detail':
        show_detail_page()

main()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏖 <b>Urban Explorer</b> - Sistem Rekomendasi Wisata Cerdas</p>
    <p>Menggunakan TF-IDF, Cosine Similarity, dan A* Road-Following Algorithm</p>
    <p>Data hotel dari OpenStreetMap • Routing dari OSRM & Custom A*</p>
</div>
""", unsafe_allow_html=True)