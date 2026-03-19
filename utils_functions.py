import networkx as nx
import pandas as pd
from networkx import community

def build_graph_from_data(edges_path, posts_path):
    """
    Loads the datasets and builds a Graph where nodes are authors 
    and edges represent interactions (comments on posts).
    """
    # Load datasets
    df_edges = pd.read_csv(edges_path)
    df_posts = pd.read_csv(posts_path)
    
    # Standardizing columns: Linking commenters (Source) to post authors (Target)
    # Based on your previous logic, we ensure the graph captures user-to-user interaction
    G = nx.from_pandas_edgelist(
        df_edges, 
        source='Source', 
        target='Target', 
        create_using=nx.Graph()
    )
    
    print(f"Graph initialized: {G.number_of_nodes()} users, {G.number_of_edges()} interactions.")
    return G

def detect_communities(G):
    """
    Implements the Louvain algorithm to identify modular communities.
    Returns a dictionary mapping {User: Community_ID}.
    """
    # Louvain algorithm for modularity optimization
    communities_list = nx.community.louvain_communities(G, seed=42)
    
    # Convert list of sets to a flat dictionary mapping
    community_mapping = {node: i for i, comm in enumerate(communities_list) for node in comm}
    
    print(f"Community Detection: Found {len(communities_list)} distinct groups.")
    return community_mapping

def calculate_centrality(G):
    """
    Calculates Betweenness (bridges) and PageRank (influencers).
    """
    print("Calculating Betweenness Centrality (using sampling for speed)...")
    # k=500 provides a good approximation for large graphs
    betweenness = nx.betweenness_centrality(G, k=500, seed=42)
    
    print("Calculating PageRank Centrality...")
    pagerank = nx.pagerank(G)
    
    return betweenness, pagerank

def identify_core_periphery(G):
    """
    Applies k-core decomposition to separate the highly engaged 
    'backbone' from casual participants.
    """
    # core_number returns the highest k-core each node belongs to
    core_numbers = nx.core_number(G)
    return core_numbers

def generate_analysis_report(G):
    """
    Executes all analyses and merges them into a single analytical DataFrame.
    """
    # Run Algorithms
    communities = detect_communities(G)
    betweenness, pagerank = calculate_centrality(G)
    core_scores = identify_core_periphery(G)
    
    # Build Report
    report = pd.DataFrame({
        'User': list(G.nodes()),
        'Community_ID': [communities.get(node) for node in G.nodes()],
        'Betweenness_Score': [betweenness.get(node) for node in G.nodes()],
        'PageRank_Score': [pagerank.get(node) for node in G.nodes()],
        'Coreness_Level': [core_scores.get(node) for node in G.nodes()]
    })
    
    # Label Core vs Periphery (Top 10% of coreness is considered the backbone)
    core_threshold = report['Coreness_Level'].quantile(0.9)
    report['Network_Role'] = report['Coreness_Level'].apply(
        lambda x: 'Core (Backbone)' if x >= core_threshold else 'Periphery'
    )
    
    return report, communities

def detect_anomalies(node_report, betweenness_threshold=0.8, pagerank_threshold=0.2):
    """
    Identifies suspicious users based on topological heuristics.
    - Potential Raid Leader: High Betweenness + High Coreness + Low PageRank.
    - Bridge/Troll: High Betweenness + Linking different communities.
    """
    
    # Normalize metrics between 0 and 1 for easier heuristic comparison
    metrics = ['Betweenness_Score', 'PageRank_Score', 'Coreness_Level']
    node_report_norm = node_report.copy()
    for m in metrics:
        node_report_norm[m] = (node_report[m] - node_report[m].min()) / (node_report[m].max() - node_report[m].min())

    alerts = []

    for index, row in node_report_norm.iterrows():
        # HEURISTIC 1: The "Raid Leader" / "Bridge Troll"
        # High Betweenness (top 10%) but relatively low PageRank (not a community pillar)
        if row['Betweenness_Score'] > node_report_norm['Betweenness_Score'].quantile(betweenness_threshold):
            if row['PageRank_Score'] < node_report_norm['PageRank_Score'].quantile(pagerank_threshold):
                alerts.append({
                    'User': row['User'],
                    'Type': 'Potential Bridge/Troll',
                    'Reason': 'High influence as a bridge but low community authority (PageRank).',
                    'Severity': 'High'
                })

        # HEURISTIC 2: Coordinated Backbone Member
        # High Coreness but very low PageRank (could be a bot or "sleeper" account)
        if row['Coreness_Level'] == 1.0 and row['PageRank_Score'] < 0.05:
            alerts.append({
                'User': row['User'],
                'Type': 'Suspicious Core Member',
                'Reason': 'Deeply embedded in the network backbone but has zero organic influence.',
                'Severity': 'Medium'
            })

    return pd.DataFrame(alerts)

def alert_community_manager(alerts_df):
    """
    Formats the alerts for a human Community Manager or DSA review.
    """
    if alerts_df.empty:
        print("✅ No suspicious patterns detected in this snapshot.")
    else:
        print(f"🚨 ALERT: {len(alerts_df)} suspicious users flagged for DSA review.")
        print("-" * 30)
        print(alerts_df.sort_values(by='Severity'))


def test_network_robustness(G, core_nodes):
    """
    Simulates the removal of core nodes and measures the impact on 
    the Largest Connected Component (LCC).
    """
    # Initial State
    initial_lcc_size = len(max(nx.connected_components(G), key=len))
    
    # Simulation: Remove identified core nodes
    G_fragmented = G.copy()
    G_fragmented.remove_nodes_from(core_nodes)
    
    # Final State
    final_lcc_size = len(max(nx.connected_components(G_fragmented), key=len)) if G_fragmented.nodes() else 0
    
    drop_percentage = ((initial_lcc_size - final_lcc_size) / initial_lcc_size) * 100
    
    print("--- Robustness Testing (Structural Pillars) ---")
    print(f"Initial LCC Size: {initial_lcc_size}")
    print(f"LCC Size after Core Removal: {final_lcc_size}")
    print(f"Network Fragmentation: {drop_percentage:.2f}% drop in connectivity.")
    
    return drop_percentage

def evaluate_filtering_impact(G, anomaly_nodes, community_mapping):
    """
    Compares global graph metrics (Modularity) before and after 
    filtering identified anomalies.
    """
    # Helper to convert mapping {node: cid} to list of sets [{n1, n2}, {n3}]
    def get_partition_list(nodes_to_keep, mapping):
        communities = {}
        for node in nodes_to_keep:
            cid = mapping[node]
            if cid not in communities: communities[cid] = set()
            communities[cid].add(node)
        return list(communities.values())

    # Initial Modularity
    initial_nodes = list(G.nodes())
    partition_initial = get_partition_list(initial_nodes, community_mapping)
    mod_initial = nx.community.modularity(G, partition_initial)
    
    # Filter Anomalies
    G_filtered = G.copy()
    G_filtered.remove_nodes_from(anomaly_nodes)
    
    # Final Modularity
    filtered_nodes = list(G_filtered.nodes())
    partition_filtered = get_partition_list(filtered_nodes, community_mapping)
    mod_filtered = nx.community.modularity(G_filtered, partition_filtered)
    
    print("\n--- Metric Evaluation (Noise Filtering) ---")
    print(f"Original Modularity: {mod_initial:.4f}")
    print(f"Filtered Modularity: {mod_filtered:.4f}")
    print(f"Improvement: {((mod_filtered - mod_initial) / mod_initial) * 100:.2f}%")
    
    return mod_initial, mod_filtered