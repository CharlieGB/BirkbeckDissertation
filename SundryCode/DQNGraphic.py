import matplotlib.pyplot as plt
import networkx as nx

# Create a graph
G = nx.DiGraph()

# Define the nodes for input, hidden, and output layers
input_neurons = ['S1', 'S2']
hidden_neurons1 = ['H1', 'H2', 'H3']
hidden_neurons2 = ['H4', 'H5', 'H6']
output_neurons = ['A1', 'A2', 'A3', 'A4']

# Adding nodes to the graph
G.add_nodes_from(input_neurons, layer='Input')
G.add_nodes_from(hidden_neurons1, layer='Hidden1')
G.add_nodes_from(hidden_neurons2, layer='Hidden2')
G.add_nodes_from(output_neurons, layer='Output')

# Adding edges with weights
edges = [(u, v) for u in input_neurons for v in hidden_neurons1] + \
        [(u, v) for u in hidden_neurons1 for v in hidden_neurons2] + \
        [(u, v) for u in hidden_neurons2 for v in output_neurons]

weights = {(u, v): i+1 for i, (u, v) in enumerate(edges)}

for (u, v), w in weights.items():
    G.add_edge(u, v, weight=w)

# Positioning the nodes in layers
pos = {}
layer_offsets = {'Input': 0, 'Hidden1': 1, 'Hidden2': 2, 'Output': 3}
layer_sizes = {'Input': len(input_neurons), 'Hidden1': len(hidden_neurons1), 'Hidden2': len(hidden_neurons2), 'Output': len(output_neurons)}

for i, node in enumerate(G.nodes()):
    layer = G.nodes[node]['layer']
    pos[node] = (layer_offsets[layer], layer_sizes[layer] - i % layer_sizes[layer])

# Plot the graph
fig, ax = plt.subplots(figsize=(12, 8))
node_colors = [layer_offsets[G.nodes[node]['layer']] for node in G.nodes()]
nx.draw(G, pos, ax=ax, with_labels=True, node_size=2000, node_color=node_colors, cmap=plt.cm.viridis, edge_color='gray')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'w={d["weight"]}' for u, v, d in G.edges(data=True)}, ax=ax)
labels = {'Input': 'Input Layer', 'Hidden1': 'Hidden Layer 1', 'Hidden2': 'Hidden Layer 2', 'Output': 'Output Layer'}
for layer, label in labels.items():
    x = layer_offsets[layer]
    y = layer_sizes[layer] + 0.5
    ax.text(x, y, label, fontsize=12, ha='center', va='center')

#plt.title("Deep Q-Network (DQN) Diagram")
plt.show(block=True)