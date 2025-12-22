import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(graph, trust_model=None, filename="network_topology.png", return_fig=False):
    """
    Visualizes the network topology.
    - Nodes are colored based on reliability/trust if provided.
    - Bad/Low Trust nodes -> Red
    - Good/High Trust nodes -> Green
    """
    fig = plt.figure(figsize=(10, 8))
    
    pos = nx.spring_layout(graph, seed=42)
    
    node_colors = []
    if trust_model:
        # If we have a trust model, use its values
        for node in graph.nodes():
            trust = trust_model.get_trust(node)
            # Interpolate Red (0) to Green (1)
            # Simple threshold for now
            if trust < 0.5:
                node_colors.append('red')
            else:
                node_colors.append('green')
    else:
        # Fallback to checking 'reliability' attribute if set in earlier simulation steps
        for node in graph.nodes():
            rel = graph.nodes[node].get('reliability', 1.0)
            if rel < 0.8:
                node_colors.append('red')
            else:
                node_colors.append('green')

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.5)
    
    # Labels
    nx.draw_networkx_labels(graph, pos)
    
    # Edge Labels (Weights/Latency)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.title("Network Topology (Green=Trusted, Red=Untrusted)")
    
    if return_fig:
        plt.close(fig) # Close global pyplot ref but return object
        return fig
        
    try:
        plt.savefig(filename)
        print(f"Network visualization saved to {filename}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    finally:
        plt.close()
