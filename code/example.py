# # import osmnx as ox

# # # Get a street network for a place name
# # G = ox.graph_from_place('Piedmont, California, USA', network_type='drive')

# # # Plot the street network
# # ox.plot_graph(G)
# import chardet

# # Read the raw bytes first
# with open('./data/12411-0015.csv', 'rb') as f:
#     raw_data = f.read()

# # Detect encoding
# result = chardet.detect(raw_data)
# encoding = result['encoding']
# print(encoding)

# # Now read the file with the detected encoding
# with open('./data/12411-0015.csv', encoding=encoding) as f:
#     content = f.read()

# import torch
# x = torch.rand(5, 3)
# print(x)

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    
    # Get the number of CUDA devices
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")

    # Print the name and properties of each CUDA device
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        device_properties = torch.cuda.get_device_properties(i)
        print(f"  Compute capability: {device_properties.major}.{device_properties.minor}")
        print(f"  Total memory: {device_properties.total_memory / (1024 ** 3):.2f} GB")
else:
    print("CUDA is not available")


