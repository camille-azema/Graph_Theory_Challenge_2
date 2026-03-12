import pandas as pd
import networkx as nx
import numpy as np
import os

def clean_and_prepare_data():
    print("Loading data...")
    df_posts = pd.read_csv('all_posts_active_subreddit.csv')
    df_com = pd.read_csv('all_comments.csv')

    print("Filtering bots and deleted accounts...")
    bots_and_deleted = ['[deleted]', 'AutoModerator', 'RemindMeBot', 'WikiTextBot', 'image_linker_bot', 'sneakpeekbot']
    
    df_posts = df_posts[~df_posts['Author'].isin(bots_and_deleted) & df_posts['Author'].notna()]
    df_com = df_com[~df_com['Author'].isin(bots_and_deleted) & df_com['Author'].notna()]

    # Convert dates and get ISO week
    print("Extracting time windows...")
    df_posts['Date & Time'] = pd.to_datetime(df_posts['Date & Time'])
    df_com['Date & Time'] = pd.to_datetime(df_com['Date & Time'])
    
    df_posts['ISOYearWeek'] = df_posts['Date & Time'].dt.strftime('%G-W%V')
    df_com['ISOYearWeek'] = df_com['Date & Time'].dt.strftime('%G-W%V')
    
    print("Building interactions...")
    # Edge type 1: Comment -> Post
    df_com_post = df_com[df_com['Parent ID'].isna()].copy()
    df_com_post = df_com_post.merge(df_posts[['Post ID', 'Author', 'Subreddit']], on='Post ID', how='inner', suffixes=('_com', '_post'))
    df_com_post = df_com_post[df_com_post['Author_com'] != df_com_post['Author_post']]
    
    # Weight: 2 for replying to post, + 0.1 for every 10 chars of content
    df_com_post['Weight'] = 2.0 + (df_com_post['Content'].str.len().fillna(0) / 100.0)
    
    edges_post = df_com_post[['Author_com', 'Author_post', 'Subreddit', 'ISOYearWeek', 'Weight']].rename(
        columns={'Author_com': 'Source', 'Author_post': 'Target'}
    )
    
    # Edge type 2: Comment -> Comment
    df_com_com = df_com[df_com['Parent ID'].notna()].copy()
    df_com_com = df_com_com.merge(df_com[['Comment ID', 'Author']], left_on='Parent ID', right_on='Comment ID', suffixes=('_com', '_parent'))
    df_com_com = df_com_com[df_com_com['Author_com'] != df_com_com['Author_parent']]
    
    # Get Subreddit from Post
    df_com_com = df_com_com.merge(df_posts[['Post ID', 'Subreddit']], on='Post ID', how='inner')
    
    # Weight: 1 for replying to comment, + 0.1 for every 10 chars
    df_com_com['Weight'] = 1.0 + (df_com_com['Content'].str.len().fillna(0) / 100.0)
    
    edges_com = df_com_com[['Author_com', 'Author_parent', 'Subreddit', 'ISOYearWeek', 'Weight']].rename(
        columns={'Author_com': 'Source', 'Author_parent': 'Target'}
    )
    
    # Consolidate all edges
    all_edges = pd.concat([edges_post, edges_com], ignore_index=True)
    
    # Aggregate edge weights for same Source->Target within the same Subreddit and Week
    all_edges = all_edges.groupby(['Source', 'Target', 'Subreddit', 'ISOYearWeek'], as_index=False)['Weight'].sum()
    
    print("Saving consolidated edges...")
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
        
    all_edges.to_csv('graphs/all_edges_consolidated.csv', index=False)
    
    print(all_edges.head())
    print("Pipeline completed.")

if __name__ == '__main__':
    clean_and_prepare_data()
