#  Smart Cairo Transport Optimizer

A comprehensive Streamlit-based web application that models and optimizes transportation infrastructure in Cairo. It integrates real-world-inspired data and interactive geospatial visualizations to aid in urban planning and transportation decision-making.

##  Features

- **Interactive Pathfinding** using Dijkstra and A* algorithms, with an emergency routing mode prioritizing medical and airport nodes.
- **Minimum Spanning Tree (MST)** generation to identify cost-efficient road networks with options to prioritize population density.
- **Bus and Metro Scheduling Optimization** based on passenger demand.
- **Road Maintenance Planning** using dynamic programming for maximum benefit within a specified budget.
- **Traffic Signal Optimization** at key intersections using congestion-aware signal timing.
- **Visual Analytics** using Folium maps to display routes, traffic flow, signal plans, and more.

##  Technologies Used

- [Streamlit](https://streamlit.io/)
- [Folium](https://python-visualization.github.io/folium/)
- Python (Heapq, Math, Collections, Dynamic Programming)
- Data Structures: Graphs, Heuristics, Priority Queues

##  Project Structure

- `streamlit_app1.py`: Main application script (contains all backend logic and Streamlit UI).
- All data (roads, neighborhoods, traffic, metro lines, etc.) are hardcoded within the script for demo purposes.

##  Getting Started

### Prerequisites

Install Python and the required packages:

```bash

pip install streamlit folium

streamlit run streamlit_app1.py




This model includes:

15 neighborhoods

10 major facilities (airports, hospitals, stadiums, etc.)

30+ road connections with dynamic traffic data

Public bus routes and metro lines with simulated demand

Tools for visual maintenance and scheduling optimization
