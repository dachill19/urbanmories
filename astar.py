import requests
import numpy as np
from geopy.distance import geodesic
import heapq
import streamlit as st
from scipy.spatial import KDTree
from functools import lru_cache
import time
from collections import defaultdict
import json

def haversine(coord1, coord2):
    """Calculate Haversine distance between two coordinates (optimized)"""
    R = 6371.0
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

class OptimizedAStarPathfinder:
    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination
        self.road_graph = defaultdict(list)
        self.all_points = {}
        self.max_nodes = 5000
        self.connection_threshold = 0.05  # 50 meters for intersections
        self.osrm_url = "http://router.project-osrm.org"  # Ganti dengan URL server OSRM Anda jika menggunakan instance sendiri
        
    def get_optimal_radius(self):
        """Calculate optimal radius based on distance"""
        direct_dist = geodesic(self.origin, self.destination).kilometers
        return min(max(direct_dist * 1.2, 3), 15)
        
    @lru_cache(maxsize=16)
    def get_road_network(self, center_lat, center_lon, radius_km):
        """Fetch road network using OSRM as primary source, with Overpass as fallback"""
        try:
            coords = f"{self.origin[1]},{self.origin[0]};{self.destination[1]},{self.destination[0]}"
            osrm_route_url = f"{self.osrm_url}/route/v1/driving/{coords}?steps=true&geometries=geojson&overview=full"
            response = requests.get(osrm_route_url, timeout=15)
            
            if response.status_code != 200:
                st.warning(f"OSRM API returned status {response.status_code}, falling back to Overpass")
                return self._get_overpass_road_network(center_lat, center_lon, radius_km)
                
            data = response.json()
            road_segments = []
            
            road_weights = {
                'motorway': 0.6, 'trunk': 0.7, 'primary': 0.8,
                'secondary': 0.9, 'tertiary': 1.0, 'residential': 1.2
            }
            
            for route in data.get('routes', []):
                for leg in route.get('legs', []):
                    for step in leg.get('steps', []):
                        coords = step['maneuver']['location']
                        geometry = step.get('geometry', {}).get('coordinates', [])
                        if len(geometry) >= 2:
                            coords = [(coord[1], coord[0]) for coord in geometry]
                            highway_type = step.get('mode', 'driving')
                            weight = road_weights.get(highway_type, 1.3)
                            total_length = sum(haversine(coords[i], coords[i+1]) 
                                             for i in range(len(coords)-1))
                            if total_length > 0.02:
                                road_segments.append({
                                    'coords': coords,
                                    'weight': weight,
                                    'id': step.get('name', 'osrm_segment'),
                                    'length': total_length
                                })
            
            st.info(f"Fetched {len(road_segments)} road segments from OSRM (radius: {radius_km:.1f}km)")
            return road_segments
            
        except Exception as e:
            st.error(f"Error fetching road network from OSRM: {str(e)}, falling back to Overpass")
            return self._get_overpass_road_network(center_lat, center_lon, radius_km)

    def _get_overpass_road_network(self, center_lat, center_lon, radius_km):
        """Fallback to Overpass API for road network"""
        try:
            overpass_url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:30];
            (
              way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"]
                  (around:{radius_km*1000},{center_lat},{center_lon});
            );
            out geom;
            """
            response = requests.get(overpass_url, params={'data': query}, timeout=25)
            
            if response.status_code != 200:
                st.warning(f"Overpass API returned status {response.status_code}")
                return []
                
            data = response.json()
            road_segments = []
            road_weights = {
                'motorway': 0.6, 'trunk': 0.7, 'primary': 0.8,
                'secondary': 0.9, 'tertiary': 1.0, 'residential': 1.2
            }
            
            for element in data.get('elements', []):
                if element.get('type') == 'way' and 'geometry' in element:
                    coords = [(node['lat'], node['lon']) for node in element['geometry']]
                    if len(coords) >= 2:
                        highway_type = element.get('tags', {}).get('highway', 'residential')
                        weight = road_weights.get(highway_type, 1.3)
                        total_length = sum(haversine(coords[i], coords[i+1]) 
                                         for i in range(len(coords)-1))
                        if total_length > 0.02:
                            road_segments.append({
                                'coords': coords,
                                'weight': weight,
                                'id': element.get('id'),
                                'length': total_length
                            })
            
            road_segments.sort(key=lambda x: (x['weight'], -x['length']))
            st.info(f"Fetched {len(road_segments)} road segments from Overpass (radius: {radius_km:.1f}km)")
            return road_segments
            
        except Exception as e:
            st.error(f"Error fetching road network from Overpass: {str(e)}")
            return []

    def build_optimized_graph(self, road_segments):
        """Optimized graph construction with better connectivity"""
        start_time = time.time()
        coords_list = []
        point_keys = []
        coord_to_point = {}
        road_endpoints = set()
        
        for road_idx, road in enumerate(road_segments):
            coords = road['coords']
            weight = road['weight']
            prev_point = None
            for i, coord in enumerate(coords):
                rounded_coord = (round(coord[0], 6), round(coord[1], 6))
                if rounded_coord not in coord_to_point:
                    point_key = rounded_coord
                    coord_to_point[rounded_coord] = point_key
                    self.all_points[point_key] = coord
                    coords_list.append(coord)
                    point_keys.append(point_key)
                else:
                    point_key = coord_to_point[rounded_coord]
                
                if i == 0 or i == len(coords) - 1:
                    road_endpoints.add(point_key)
                
                if prev_point is not None:
                    distance = haversine(self.all_points[prev_point], coord)
                    if distance > 0.001:
                        self.road_graph[prev_point].append({
                            'point': point_key,
                            'distance': distance,
                            'weight': weight,
                            'road_id': road['id']
                        })
                        self.road_graph[point_key].append({
                            'point': prev_point,
                            'distance': distance,
                            'weight': weight,
                            'road_id': road['id']
                        })
                
                prev_point = point_key
                if len(coords_list) >= self.max_nodes:
                    break
            
            if len(coords_list) >= self.max_nodes:
                break
        
        if len(coords_list) > 1:
            tree = KDTree(coords_list)
            threshold_deg = self.connection_threshold / 111.0
            pairs = tree.query_pairs(threshold_deg, output_type='ndarray')
            
            intersection_count = 0
            for i, j in pairs:
                if i < len(point_keys) and j < len(point_keys):
                    point1 = point_keys[i]
                    point2 = point_keys[j]
                    if point1 in road_endpoints or point2 in road_endpoints:
                        already_connected = any(
                            neighbor['point'] == point2 
                            for neighbor in self.road_graph[point1]
                        )
                        if not already_connected:
                            distance = haversine(coords_list[i], coords_list[j])
                            if distance < self.connection_threshold:
                                intersection_weight = 1.5
                                self.road_graph[point1].append({
                                    'point': point2,
                                    'distance': distance,
                                    'weight': intersection_weight,
                                    'road_id': 'intersection'
                                })
                                self.road_graph[point2].append({
                                    'point': point1,
                                    'distance': distance,
                                    'weight': intersection_weight,
                                    'road_id': 'intersection'
                                })
                                intersection_count += 1
            
            st.info(f"Added {intersection_count} intersection connections")
        
        for point in self.road_graph:
            connections = self.road_graph[point]
            connections.sort(key=lambda x: x['distance'] * x['weight'])
            self.road_graph[point] = connections[:10]
        
        build_time = time.time() - start_time
        st.info(f"Graph built in {build_time:.2f}s with {len(self.all_points)} nodes")
        return build_time < 20

    def find_nearest_points(self, target_point, k=3):
        """Find k nearest graph points with distance check"""
        if not self.all_points:
            return []
            
        coords = np.array(list(self.all_points.values()))
        target_array = np.array([target_point])
        
        tree = KDTree(coords)
        distances, indices = tree.query(target_array, k=min(k, len(coords)))
        
        if np.isscalar(indices):
            indices = [indices]
            distances = [distances]
        else:
            indices = indices[0]
            distances = distances[0]
        
        point_keys = list(self.all_points.keys())
        results = []
        
        for idx, dist in zip(indices, distances):
            if dist < 0.1:
                results.append((point_keys[idx], dist))
        
        return results

    def enhanced_a_star(self):
        """Enhanced A* using OSRM only for road network data"""
        start_time = time.time()
        radius = self.get_optimal_radius()
        center_lat = (self.origin[0] + self.destination[0]) / 2
        center_lon = (self.origin[1] + self.destination[1]) / 2
        
        road_segments = self.get_road_network(center_lat, center_lon, radius)
        
        if not road_segments:
            return self._fallback_result("No road data available")
        
        if not self.build_optimized_graph(road_segments):
            return self._fallback_result("Graph construction failed or timed out")
        
        start_candidates = self.find_nearest_points(self.origin, k=3)
        end_candidates = self.find_nearest_points(self.destination, k=3)
        
        if not start_candidates or not end_candidates:
            return self._fallback_result("Could not find suitable start/end points")
        
        st.info(f"Found {len(start_candidates)} start and {len(end_candidates)} end candidates")
        
        best_result = None
        best_distance = float('inf')
        
        for start_point, start_dist in start_candidates[:2]:
            for end_point, end_dist in end_candidates[:2]:
                if time.time() - start_time > 25:
                    break
                
                result = self._run_a_star(start_point, end_point, start_dist, end_dist)
                
                if result and result['distance'] < best_distance:
                    best_distance = result['distance']
                    best_result = result
        
        if best_result:
            best_result['method'] = 'Pure A* with OSRM Network'
            best_result['note'] = f'Optimized route using {len(road_segments)} OSRM road segments, computed with A*'
            return best_result
        else:
            return self._fallback_result("No valid path found with A*")

    def _run_a_star(self, start_point, end_point, start_dist, end_dist):
        """Run A* algorithm between two specific points"""
        open_set = [(0, start_point, 0, [])]
        g_score = {start_point: 0}
        closed_set = set()
        iteration = 0
        max_iterations = min(self.max_nodes, 3000)
        
        dest_coord = self.all_points[end_point]
        
        while open_set and iteration < max_iterations:
            iteration += 1
            current_f, current, current_g, current_path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            new_path = current_path + [current]
            
            if current == end_point:
                final_path = []
                if start_dist > 0.01:
                    final_path.append(self.origin)
                
                for point in new_path:
                    final_path.append(self.all_points[point])
                
                if end_dist > 0.01:
                    final_path.append(self.destination)
                
                total_distance = 0
                for i in range(len(final_path) - 1):
                    total_distance += haversine(final_path[i], final_path[i + 1])
                
                return {
                    'path': final_path,
                    'distance': total_distance,
                    'visited_nodes': final_path[::max(1, len(final_path)//30)],
                    'iterations': iteration,
                    'start_dist': start_dist,
                    'end_dist': end_dist
                }
            
            current_coord = self.all_points[current]
            
            for neighbor_info in self.road_graph.get(current, []):
                neighbor = neighbor_info['point']
                
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = (current_g + 
                                   neighbor_info['distance'] * neighbor_info['weight'])
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    neighbor_coord = self.all_points[neighbor]
                    h_score = haversine(neighbor_coord, dest_coord)
                    direction_penalty = 0
                    if len(new_path) >= 2:
                        prev_coord = self.all_points[new_path[-2]]
                        direction_penalty = 0.01
                    
                    f_score = tentative_g_score + h_score + direction_penalty
                    heapq.heappush(open_set, (f_score, neighbor, tentative_g_score, new_path))
        
        return None

    def _fallback_result(self, reason):
        """Return direct route as fallback without OSRM"""
        return {
            'path': [self.origin, self.destination],
            'distance': geodesic(self.origin, self.destination).kilometers,
            'visited_nodes': [self.origin, self.destination],
            'iterations': 1,
            'method': 'Direct',
            'note': reason + ' - using direct route (no OSRM fallback)'
        }

def calculate_optimized_astar_path(origin, destination, method='enhanced'):
    """
    Calculate path using optimized A* algorithm with OSRM for network data only
    
    Args:
        origin: (lat, lon) tuple for start point
        destination: (lat, lon) tuple for end point  
        method: 'enhanced' or 'simple'
    
    Returns:
        Dictionary with path information
    """
    pathfinder = OptimizedAStarPathfinder(origin, destination)
    return pathfinder.enhanced_a_star()
    