import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Define the city and center point for Trier
place_name = "Trier, Germany"
city_center = ox.geocoder.geocode(place_name)  # Get the lat/lon of city center

# Define the network type and distance window (1000 meters)
network_type = 'all'  # Can be 'drive', 'walk', etc.
distance = 1000  # Meters from the city center

# Get the street network within 1000 meters from the city center
G = ox.graph_from_point(city_center, dist=distance, network_type=network_type)

# Plot the street network with white background and black edges
fig, ax = ox.plot_graph(
    G, 
    bgcolor='white',       # Set background color to white
    node_color='black',    # Set node color to black
    node_size=5,           # Set node size to 0 to hide them
    edge_color='black',    # Set edge color to black
    edge_linewidth=0.5,    # Adjust edge line width
    show=False,            # Prevent automatic display
    close=False            # Keep the plot open for customization
)

# Set the border (spines) to black
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')

# Set spine thickness (optional)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

plt.title('Street Network of Trier', color='black')  # Optional title color
plt.show()
