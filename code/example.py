# import osmnx as ox

# # Get a street network for a place name
# G = ox.graph_from_place('Piedmont, California, USA', network_type='drive')

# # Plot the street network
# ox.plot_graph(G)
import chardet

# Read the raw bytes first
with open('./data/12411-0015.csv', 'rb') as f:
    raw_data = f.read()

# Detect encoding
result = chardet.detect(raw_data)
encoding = result['encoding']
print(encoding)

# Now read the file with the detected encoding
with open('./data/12411-0015.csv', encoding=encoding) as f:
    content = f.read()


