import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Define the city and center point for Trier
place_name = "Trier, Germany"
city_center = ox.geocoder.geocode(place_name)  # Get the lat/lon of city center

# Define the network type and distance window (1000 meters)
network_type = 'walk'  # Can be 'drive', 'walk', etc.
distance = 1000  # Meters from the city center

# Get the street network within 1000 meters from the city center
G = ox.graph_from_point(city_center, dist=distance, network_type=network_type)

# Plot the street network with white background and black edges
fig, ax = ox.plot_graph(
    G, 
    bgcolor='white',       # Set background color to white
    node_color='black',    # Set node color to black
    node_size=8,           # Adjust node size if you want them visible
    edge_color='black',    # Set edge color to black
    edge_linewidth=0.6    # Adjust edge width
)

plt.title('Street Network of Trier')
plt.show()


# Set the plot size
# fig, ax = plt.subplots(figsize=(12, 12))

# # Get node positions (lat, lon)
# pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

# # Draw the graph
# nx.draw(
#     G, 
#     pos, 
#     ax=ax, 
#     node_size=10, 
#     node_color='blue', 
#     edge_color='gray', 
#     with_labels=False
# )

# # Add node latitude and longitude labels
# for node, data in G.nodes(data=True):
#     lat = data['y']
#     lon = data['x']
#     ax.text(lon, lat, f"({lat:.4f}, {lon:.4f})", fontsize=2.5, color='black')

# # Add edge lengths (distances) as labels on the edges
# for u, v, data in G.edges(data=True):
#     if 'length' in data:
#         # Get the midpoint for the edge
#         x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
#         x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
#         mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

#         # Add text with street distance
#         distance = data['length']
#         ax.text(mid_x, mid_y, f"{distance:.1f} m", fontsize=8, color='green')

# # Set labels and title
# plt.title(f"Street Network within 250m of Trier City Center")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# # Show plot
# plt.show()
