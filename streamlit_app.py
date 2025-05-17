import heapq
import math
from typing import Union
import folium
import streamlit as st
from streamlit.components.v1 import html


# ============ Data ============
def load_all_data():
    neighborhoods = {1: {"name": "Maadi", "pop": 250000, "type": "Residential", "coord": (31.25, 29.96)},
                     2: {"name": "Nasr City", "pop": 500000, "type": "Mixed", "coord": (31.34, 30.06)},
                     3: {"name": "Downtown Cairo", "pop": 100000, "type": "Business", "coord": (31.24, 30.04)},
                     4: {"name": "New Cairo", "pop": 300000, "type": "Residential", "coord": (31.47, 30.03)},
                     5: {"name": "Heliopolis", "pop": 200000, "type": "Mixed", "coord": (31.32, 30.09)},
                     6: {"name": "Zamalek", "pop": 50000, "type": "Residential", "coord": (31.22, 30.06)},
                     7: {"name": "6th October City", "pop": 400000, "type": "Mixed", "coord": (30.98, 29.93)},
                     8: {"name": "Giza", "pop": 550000, "type": "Mixed", "coord": (31.21, 29.99)},
                     9: {"name": "Mohandessin", "pop": 180000, "type": "Business", "coord": (31.20, 30.05)},
                     10: {"name": "Dokki", "pop": 220000, "type": "Mixed", "coord": (31.21, 30.03)},
                     11: {"name": "Shubra", "pop": 450000, "type": "Residential", "coord": (31.24, 30.11)},
                     12: {"name": "Helwan", "pop": 350000, "type": "Industrial", "coord": (31.33, 29.85)},
                     13: {"name": "New Administrative Capital", "pop": 50000, "type": "Government",
                          "coord": (31.80, 30.02)},
                     14: {"name": "Al Rehab", "pop": 120000, "type": "Residential", "coord": (31.49, 30.06)},
                     15: {"name": "Sheikh Zayed", "pop": 150000, "type": "Residential", "coord": (30.94, 30.01)}}

    facilities = {"F1": {"name": "Cairo International Airport", "type": "Airport", "coord": (31.41, 30.11)},
                  "F2": {"name": "Ramses Railway Station", "type": "Transit Hub", "coord": (31.25, 30.06)},
                  "F3": {"name": "Cairo University", "type": "Education", "coord": (31.21, 30.03)},
                  "F4": {"name": "Al-Azhar University", "type": "Education", "coord": (31.26, 30.05)},
                  "F5": {"name": "Egyptian Museum", "type": "Tourism", "coord": (31.23, 30.05)},
                  "F6": {"name": "Cairo International Stadium", "type": "Sports", "coord": (31.30, 30.07)},
                  "F7": {"name": "Smart Village", "type": "Business", "coord": (30.97, 30.07)},
                  "F8": {"name": "Cairo Festival City", "type": "Commercial", "coord": (31.40, 30.03)},
                  "F9": {"name": "Qasr El Aini Hospital", "type": "Medical", "coord": (31.23, 30.03)},
                  "F10": {"name": "Maadi Military Hospital", "type": "Medical", "coord": (31.25, 29.95)}}

    roads = [(1, 3, 8.5, 3000, 7), (1, 8, 6.2, 2500, 6), (2, 3, 5.9, 2800, 8), (2, 5, 4.0, 3200, 9),
             (3, 5, 6.1, 3500, 7), (3, 6, 3.2, 2000, 8), (3, 9, 4.5, 2600, 6), (3, 10, 3.8, 2400, 7),
             (4, 2, 15.2, 3800, 9), (4, 14, 5.3, 3000, 10), (5, 11, 7.9, 3100, 7), (6, 9, 2.2, 1800, 8),
             (7, 8, 24.5, 3500, 8), (7, 15, 9.8, 3000, 9), (8, 10, 3.3, 2200, 7), (8, 12, 14.8, 2600, 5),
             (9, 10, 2.1, 1900, 7), (10, 11, 8.7, 2400, 6), (11, "F2", 3.6, 2200, 7), (12, 1, 12.7, 2800, 6),
             (13, 4, 45.0, 4000, 10), (14, 13, 35.5, 3800, 9), (15, 7, 9.8, 3000, 9), ("F1", 5, 7.5, 3500, 9),
             ("F1", 2, 9.2, 3200, 8), ("F2", 3, 2.5, 2000, 7), ("F7", 15, 8.3, 2800, 8), ("F8", 4, 6.1, 3000, 9)]

    potential_roads = [(1, 4, 22.8, 4000, 450), (1, 14, 25.3, 3800, 500), (2, 13, 48.2, 4500, 950),
                       (3, 13, 56.7, 4500, 1100), (5, 4, 16.8, 3500, 320), (6, 8, 7.5, 2500, 150),
                       (7, 13, 82.3, 4000, 1600), (9, 11, 6.9, 2800, 140), (10, "F7", 27.4, 3200, 550),
                       (11, 13, 62.1, 4200, 1250), (12, 14, 30.5, 3600, 610), (14, 5, 18.2, 3300, 360),
                       (15, 9, 22.7, 3000, 450), ("F1", 13, 40.2, 4000, 800), ("F7", 9, 26.8, 3200, 540)]

    traffic_flow = {"1-3": {"morning": 2800, "afternoon": 1500, "evening": 2600, "night": 800},
                    "1-8": {"morning": 2200, "afternoon": 1200, "evening": 2100, "night": 600},
                    "2-3": {"morning": 2700, "afternoon": 1400, "evening": 2500, "night": 700},
                    "2-5": {"morning": 3000, "afternoon": 1600, "evening": 2800, "night": 650},
                    "3-5": {"morning": 3200, "afternoon": 1700, "evening": 3100, "night": 800},
                    "3-6": {"morning": 1800, "afternoon": 1400, "evening": 1900, "night": 500},
                    "3-9": {"morning": 2400, "afternoon": 1300, "evening": 2200, "night": 550},
                    "3-10": {"morning": 2300, "afternoon": 1200, "evening": 2100, "night": 500},
                    "4-2": {"morning": 3600, "afternoon": 1800, "evening": 3300, "night": 750},
                    "4-14": {"morning": 2800, "afternoon": 1600, "evening": 2600, "night": 600},
                    "5-11": {"morning": 2900, "afternoon": 1500, "evening": 2700, "night": 650},
                    "6-9": {"morning": 1700, "afternoon": 1300, "evening": 1800, "night": 450},
                    "7-8": {"morning": 3200, "afternoon": 1700, "evening": 3000, "night": 700},
                    "7-15": {"morning": 2800, "afternoon": 1500, "evening": 2600, "night": 600},
                    "8-10": {"morning": 2000, "afternoon": 1100, "evening": 1900, "night": 450},
                    "8-12": {"morning": 2400, "afternoon": 1300, "evening": 2200, "night": 500},
                    "9-10": {"morning": 1800, "afternoon": 1200, "evening": 1700, "night": 400},
                    "10-11": {"morning": 2200, "afternoon": 1300, "evening": 2100, "night": 500},
                    "11-F2": {"morning": 2100, "afternoon": 1200, "evening": 2000, "night": 450},
                    "12-1": {"morning": 2600, "afternoon": 1400, "evening": 2400, "night": 550},
                    "13-4": {"morning": 3800, "afternoon": 2000, "evening": 3500, "night": 800},
                    "14-13": {"morning": 3600, "afternoon": 1900, "evening": 3300, "night": 750},
                    "15-7": {"morning": 2800, "afternoon": 1500, "evening": 2600, "night": 600},
                    "F1-5": {"morning": 3300, "afternoon": 2200, "evening": 3100, "night": 1200},
                    "F1-2": {"morning": 3000, "afternoon": 2000, "evening": 2800, "night": 1100},
                    "F2-3": {"morning": 1900, "afternoon": 1600, "evening": 1800, "night": 900},
                    "F7-15": {"morning": 2600, "afternoon": 1500, "evening": 2400, "night": 550},
                    "F8-4": {"morning": 2800, "afternoon": 1600, "evening": 2600, "night": 600}}

    bus_routes = {"B1": {"stops": [1, 3, 6, 9], "buses": 25, "passengers": 35000},
                  "B2": {"stops": [7, 15, 8, 10, 3], "buses": 30, "passengers": 42000},
                  "B3": {"stops": [2, 5, "F1"], "buses": 20, "passengers": 28000},
                  "B4": {"stops": [4, 14, 2, 3], "buses": 22, "passengers": 31000},
                  "B5": {"stops": [8, 12, 1], "buses": 18, "passengers": 25000},
                  "B6": {"stops": [11, 5, 2], "buses": 24, "passengers": 33000},
                  "B7": {"stops": [13, 4, 14], "buses": 15, "passengers": 21000},
                  "B8": {"stops": ["F7", 15, 7], "buses": 12, "passengers": 17000},
                  "B9": {"stops": [1, 8, 10, 9, 6], "buses": 28, "passengers": 39000},
                  "B10": {"stops": ["F8", 4, 2, 5], "buses": 20, "passengers": 28000}}

    public_demand = {(3, 5): 15000, (1, 3): 12000, (2, 3): 18000, ("F2", 11): 25000, ("F1", 3): 20000, (7, 3): 14000,
                     (4, 3): 16000, (8, 3): 22000, (3, 9): 13000, (5, 2): 17000, (11, 3): 24000, (12, 3): 11000,
                     (1, 8): 9000, (7, "F7"): 18000, (4, "F8"): 12000, (13, 3): 8000, (14, 4): 7000}

    return neighborhoods, facilities, roads, potential_roads, traffic_flow, bus_routes, public_demand


# ============ Graph Structure ============
class CityGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, nid, meta):
        self.nodes[nid] = meta

    def add_edge(self, u, v, dist, cap, cond):
        self.edges.append((dist, u, v, {"capacity": cap, "condition": cond}))

    def get_neighbors(self, u):
        neighbors = []
        for dist, src, dst, _ in self.edges:
            if src == u:
                neighbors.append((dst, dist))
            elif dst == u:
                neighbors.append((src, dist))
        return neighbors

    def heuristic(self, a, b):
        x1, y1 = self.nodes[a]['coord']
        x2, y2 = self.nodes[b]['coord']
        return math.hypot(x1 - x2, y1 - y2)


# ============ Algorithms ============
def dijkstra(graph, start, end):
    dist = {node: float('inf') for node in graph.nodes}
    prev = {}
    dist[start] = 0
    pq: list[tuple[float, Union[int, str]]] = [(0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if node == end:
            path = []
            while node in prev:
                path.append(node)
                node = prev[node]
            path.append(start)
            return d, path[::-1]
        for neighbor, weight in graph.get_neighbors(node):
            if d + weight < dist[neighbor]:
                dist[neighbor] = d + weight
                prev[neighbor] = node
                heapq.heappush(pq, (dist[neighbor], neighbor))
    return float('inf'), []


def astar(graph, start, end):
    open_set = [(graph.heuristic(start, end), 0, start)]
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    prev = {}
    while open_set:
        _, cost, curr = heapq.heappop(open_set)
        if curr == end:
            path = []
            while curr in prev:
                path.append(curr)
                curr = prev[curr]
            path.append(start)
            return cost, path[::-1]
        for neighbor, weight in graph.get_neighbors(curr):
            g = cost + weight
            if g < g_score[neighbor]:
                g_score[neighbor] = g
                prev[neighbor] = curr
                heapq.heappush(open_set, (g + graph.heuristic(neighbor, end), g, neighbor))
    return float('inf'), []


def run_mst(graph):
    parent = {node: node for node in graph.nodes}

    def find(l):
        if parent[l] != l:
            parent[l] = find(parent[l])
        return parent[l]

    def union(k, p):
        pu, pv = find(k), find(p)
        if pu != pv:
            parent[pu] = pv
            return True
        return False

    mst = []
    for dist, u, v, _ in sorted(graph.edges, key=lambda x: int(x[0])):
        if union(u, v):
            mst.append((u, v))

    return mst


def schedule_buses(demand_list, total_buses):
    n = len(demand_list)
    demand_list = sorted(demand_list, key=lambda x: x[1], reverse=True)
    allocation = [1] * n
    buses_remaining = total_buses - n
    while buses_remaining > 0:
        best_index = -1
        best_improvement = -1
        for i in range(n):
            current = demand_list[i][1] / allocation[i]
            potential = demand_list[i][1] / (allocation[i] + 1)
            improvement = current - potential
            if improvement > best_improvement:
                best_improvement = improvement
                best_index = i
        allocation[best_index] += 1
        buses_remaining -= 1
    avg_wait = sum(demand_list[i][1] / allocation[i] for i in range(n)) / n
    return avg_wait


def road_maintenance(roads, budget):
    n = len(roads)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost, benefit = roads[i - 1]
        for b in range(budget + 1):
            dp[i][b] = dp[i - 1][b]
            if b >= cost:
                dp[i][b] = max(dp[i][b], dp[i - 1][b - cost] + benefit)
    sel, b = [], budget
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            sel.append(i - 1)
            b -= roads[i - 1][0]
    return dp[n][budget], sel[::-1]


# ============ Visualization ============
def draw_folium_map(graph, edges=None, highlight_nodes=None):
    # Center map around Cairo
    cairo_center = [30.05, 31.25]
    m = folium.Map(location=cairo_center, zoom_start=11)
    # Add all nodes
    for nid, meta in graph.nodes.items():
        lat, lon = meta["coord"][1], meta["coord"][0]  # folium uses (lat, lon)
        label = f"{meta['name']} ({nid})"
        color = "blue" if isinstance(nid, int) else "green"
        folium.CircleMarker(location=(lat, lon), radius=5, color=color, fill=True, fill_opacity=0.8,
                            popup=label).add_to(m)
    # Draw edges if given
    if edges:
        for u, v in edges:
            u_lat, u_lon = graph.nodes[u]["coord"][1], graph.nodes[u]["coord"][0]
            v_lat, v_lon = graph.nodes[v]["coord"][1], graph.nodes[v]["coord"][0]
            folium.PolyLine(locations=[(u_lat, u_lon), (v_lat, v_lon)], color='green', weight=3, opacity=0.7).add_to(m)
    # Draw path if specified
    if highlight_nodes:
        latlon_path = [(graph.nodes[nid]['coord'][1], graph.nodes[nid]['coord'][0]) for nid in highlight_nodes]
        folium.PolyLine(locations=latlon_path, color='blue', weight=4, opacity=0.9).add_to(m)
    # Render folium map in Streamlit
    html(m._repr_html_(), height=600, scrolling=False)


# ============ Streamlit Interface ============
def main():
    st.set_page_config(layout="wide")
    st.title("🚌 Smart Cairo Transport Optimizer")
    neighborhoods, facilities, roads, potential_roads, traffic_flow, bus_routes, public_demand = load_all_data()
    graph = CityGraph()
    # Add neighborhood nodes
    for nid, meta in neighborhoods.items():
        graph.add_node(nid, meta)
    # Add facility nodes (this line is currently missing)
    for fid, meta in facilities.items():
        graph.add_node(fid, meta)
    # Add edges
    for u, v, dist, cap, cond in roads:
        graph.add_edge(u, v, dist, cap, cond)
    # Node dictionary for UI selection (limit to neighborhood names only)
    nodes = {nid: data["name"] for nid, data in neighborhoods.items()}
    tab1, tab2, tab3 = st.tabs(["🚦 Pathfinding", "🌲 MST", "📊 Optimization"])
    with tab1:
        st.header("🚗 Pathfinding")
        start = st.selectbox("Start", nodes.keys(), format_func=lambda x: nodes[x])
        end = st.selectbox("End", nodes.keys(), index=2, format_func=lambda x: nodes[x])
        algo = st.radio("Algorithm", ["Dijkstra", "A*"])
        if st.button("Run Path"):
            cost, path = dijkstra(graph, start, end) if algo == "Dijkstra" else astar(graph, start, end)
            st.success(f"{algo} Cost: {cost:.2f}, Path: {path}")
            draw_folium_map(graph, highlight_nodes=path)
    with tab2:
        st.header("🌐 Minimum Spanning Tree")
        if st.button("Show MST"):
            mst = run_mst(graph)
            st.subheader("🗺 MST on Interactive Map")
            draw_folium_map(graph, edges=mst)
    with tab3:
        st.header("🚌 Bus Scheduling")
        total_buses = st.slider("Total Available Buses", 50, 200, 100)
        demand_list = [(rid, r["passengers"]) for rid, r in bus_routes.items()]
        if st.button("Schedule Buses"):
            wait = schedule_buses(demand_list, total_buses)
            st.info(f"Minimized Avg Waiting Time: {wait:.2f}")
        st.header("🛠 Maintenance Planning")
        budget = st.slider("Maintenance Budget", 500, 5000, 1500)
        # Use real road data with condition ≤ 6
        roads_data = [(int(dist * 10), (10 - cond) * 100) for u, v, dist, cap, cond in roads if cond <= 6]
        benefit, selected = road_maintenance(roads_data, budget)
        st.success(f"Selected roads: {selected}, Benefit Score: {benefit}")


if __name__ == "__main__":
    main()
