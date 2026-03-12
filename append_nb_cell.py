import nbformat as nbf
import os

nb_path = '1_data_preparation.ipynb'
if not os.path.exists(nb_path):
    print(f"File {nb_path} not found.")
    exit(1)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

code = """import pandas as pd
import networkx as nx

# Load consolidated edges
edges_df = pd.read_csv('graphs/all_edges_consolidated.csv')

def create_weekly_subreddit_graphs(df):
    graphs = {}
    for (subreddit, year_week), group in df.groupby(['Subreddit', 'ISOYearWeek']):
        if subreddit not in graphs:
            graphs[subreddit] = {}
            
        G = nx.DiGraph()
        for _, row in group.iterrows():
            # Source replied to Target
            G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
            
        graphs[subreddit][year_week] = G
    return graphs

weekly_graphs = create_weekly_subreddit_graphs(edges_df)

def get_top_n_edges(graph, n=10):
    edges = graph.edges(data=True)
    sorted_edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
    return sorted_edges[:n]

# Example top 10 edges for a specific graph (e.g. ClimateActionPlan, first week available)
subreddit_example = 'ClimateActionPlan'
if subreddit_example in weekly_graphs:
    week_example = list(weekly_graphs[subreddit_example].keys())[0]
    example_graph = weekly_graphs[subreddit_example][week_example]
    
    print(f"Top 10 edges for {subreddit_example} in week {week_example}:")
    for edge in get_top_n_edges(example_graph, n=10):
        print(edge)
"""

cell = nbf.v4.new_code_cell(code)
nb['cells'].append(cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Cell appended successfully.")
