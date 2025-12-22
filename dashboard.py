import streamlit as st
import simpy
import networkx as nx
import random
import matplotlib.pyplot as plt
from network_sim import NetworkSimulation
from trust_model import TrustModel
from routing import IntelligentRouting, ShortestPathRouting, RLRouting, RIPRouting
from rl_agent import QLearningAgent
from visualization import visualize_network
import pandas as pd

st.set_page_config(page_title="Intelligent Routing Dashboard", layout="wide")

st.title("Network Routing Simulation Dashboard")

# Initialize Session State
if 'env' not in st.session_state:
    st.session_state.env = simpy.Environment()
    st.session_state.net_sim = NetworkSimulation(st.session_state.env)
    st.session_state.net_sim.create_topology(num_nodes=15, connectivity=0.2)
    
    # Pre-seed some bad nodes for demo purposes
    st.session_state.net_sim.graph.nodes[3]['reliability'] = 0.5
    st.session_state.net_sim.graph.nodes[7]['reliability'] = 0.6
    
    st.session_state.trust_model = TrustModel()
    
    # Initialize RL Agent
    st.session_state.rl_agent = QLearningAgent(list(st.session_state.net_sim.graph.nodes))
    
    # Default Routing
    st.session_state.routing_algo_name = "Intelligent (Trust)"
    st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    
    st.session_state.packet_stats = []
    st.session_state.time = 0

# Sidebar Controls
st.sidebar.header("Simulation Controls")

# Algorithm Selection
algo_option = st.sidebar.selectbox(
    "Routing Protocol", 
    ["Standard OSPF (Latency)", "RIP (Hop Count)", "Intelligent (Trust)", "Q-Learning (AI)"]
)

# Handle Algorithm Change
if algo_option != st.session_state.routing_algo_name:
    st.session_state.routing_algo_name = algo_option
    if algo_option == "Standard OSPF (Latency)":
        st.session_state.routing = ShortestPathRouting(st.session_state.net_sim.graph)
    elif algo_option == "RIP (Hop Count)":
        st.session_state.routing = RIPRouting(st.session_state.net_sim.graph)
    elif algo_option == "Intelligent (Trust)":
        st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    elif algo_option == "Q-Learning (AI)":
        st.session_state.routing = RLRouting(st.session_state.net_sim.graph, st.session_state.rl_agent)
    
    st.toast(f"Switched to {algo_option}")

# Node Reliability Controls
st.sidebar.subheader("Node Reliability")
nodes = list(st.session_state.net_sim.graph.nodes)
selected_node = st.sidebar.selectbox("Select Node to Modify", nodes)
reliability = st.sidebar.slider(f"Reliability of Node {selected_node}", 0.0, 1.0, 
                                st.session_state.net_sim.graph.nodes[selected_node].get('reliability', 1.0))

if st.sidebar.button("Update Node"):
    st.session_state.net_sim.graph.nodes[selected_node]['reliability'] = reliability
    st.success(f"Node {selected_node} reliability set to {reliability}")

# Simulation Step
st.sidebar.subheader("Run Simulation")
steps = st.sidebar.number_input("Packets to Send", min_value=1, value=10)

def run_step(num_packets):
    env = st.session_state.env
    net_sim = st.session_state.net_sim
    routing = st.session_state.routing
    trust_model = st.session_state.trust_model
    
    for _ in range(num_packets):
        src, dst = random.sample(nodes, 2)
        path = routing.find_path(src, dst)
        
        status = "No Path"
        path_latency = 0
        
        if path:
            # We must use base simulate logic or patch it. 
            # Replicating logic here for transparency in dashboard
            success = True
            
            # Calculate Latency
            # path_latency = sum(net_sim.graph[u][v].get('weight', 1) for u, v in zip(path[:-1], path[1:]))
            
            for i in range(len(path) - 1):
                 u, v = path[i], path[i+1]
                 # Congestion/Reliability check
                 rel = net_sim.graph.nodes[v].get('reliability', 1.0)
                 if random.random() > rel:
                     success = False
                     trust_model.update_trust(v, False)
                     break
                 else:
                     trust_model.update_trust(v, True)
            
            # RL Feedback Loop
            if isinstance(routing, RLRouting):
                # Simple reward: -1 per hop (latency approximation) + penalty for drop
                reward = -len(path) if success else -100
                 # Update Q-Values along the path
                for i in range(len(path) - 1):
                    u_node = path[i]
                    v_node = path[i+1]
                    next_neighbors = list(net_sim.graph.neighbors(v_node)) if v_node in net_sim.graph else []
                    routing.agent.learn(u_node, v_node, reward, v_node, next_neighbors)

            status = "Success" if success else "Dropped"
        
        st.session_state.packet_stats.append({
            "Time": st.session_state.time,
            "Algorithm": st.session_state.routing_algo_name,
            "Source": src,
            "Dest": dst,
            "Path": str(path),
            "Status": status
        })
        st.session_state.time += 1

if st.sidebar.button("Simulation Step"):
    run_step(steps)

# Layout
col1, col2 = st.columns([2, 1])

# Visualize Controls
view_mode = st.radio("Visualization Mode", ["Ground Truth (Reliability)", "Agent Perception (Trust Score)"])

with col1:
    st.subheader(f"Network Topology ({view_mode})")
    
    # If Ground Truth, pass trust_model=None so it uses 'reliability' attribute
    tm_to_use = st.session_state.trust_model if view_mode == "Agent Perception (Trust Score)" else None
    
    # Visualize
    fig = visualize_network(st.session_state.net_sim.graph, 
                            trust_model=tm_to_use, 
                            return_fig=True)
    st.pyplot(fig)

with col2:
    st.subheader("Recent Traffic")
    if st.session_state.packet_stats:
        df = pd.DataFrame(st.session_state.packet_stats[-10:]) # Last 10
        st.dataframe(df)
        
        # Stats
        all_df = pd.DataFrame(st.session_state.packet_stats)
        pdr = (all_df[all_df['Status'] == 'Success'].shape[0] / all_df.shape[0]) * 100
        st.metric("Packet Delivery Ratio", f"{pdr:.2f}%")
    else:
        st.info("Run simulation to see stats.")

# Trust Table
st.subheader("Node Trust Scores")
trust_data = {n: st.session_state.trust_model.get_trust(n) for n in nodes}
st.bar_chart(trust_data)
