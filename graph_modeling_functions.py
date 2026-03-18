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
    return G, df_posts

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

def generate_analysis_report(G, df_posts):
    """
    Executes all analyses and merges them into a single analytical DataFrame.
    """
    # 1. Run Algorithms
    communities = detect_communities(G)
    betweenness, pagerank = calculate_centrality(G)
    core_scores = identify_core_periphery(G)
    
    # 2. Build Report
    report = pd.DataFrame({
        'User': list(G.nodes()),
        'Community_ID': [communities.get(node) for node in G.nodes()],
        'Betweenness_Score': [betweenness.get(node) for node in G.nodes()],
        'PageRank_Score': [pagerank.get(node) for node in G.nodes()],
        'Coreness_Level': [core_scores.get(node) for node in G.nodes()]
    })
    
    # 3. Label Core vs Periphery (Top 10% of coreness is considered the backbone)
    core_threshold = report['Coreness_Level'].quantile(0.9)
    report['Network_Role'] = report['Coreness_Level'].apply(
        lambda x: 'Core (Backbone)' if x >= core_threshold else 'Periphery'
    )
    
    return report