import nbformat as nbf
import os

nb_path = '1_data_preparation.ipynb'
if not os.path.exists(nb_path):
    print(f"File {nb_path} not found.")
    exit(1)

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Define the new content of the first cell
new_source = [
    "import pandas as pd\n",
    "import networkx as nx\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "def clean_and_prepare_data():\n",
    "    print(\"Loading data...\")\n",
    "    df_posts = pd.read_csv('all_posts_active_subreddit.csv')\n",
    "    df_com = pd.read_csv('all_comments.csv')\n",
    "\n",
    "    print(\"Filtering bots and deleted accounts...\")\n",
    "    bots_and_deleted = ['[deleted]', 'AutoModerator', 'RemindMeBot', 'WikiTextBot', 'image_linker_bot', 'sneakpeekbot']\n",
    "    \n",
    "    # Statistics before cleaning\n",
    "    posts_before = len(df_posts)\n",
    "    com_before = len(df_com)\n",
    "    unique_authors_before = set(df_posts['Author'].unique()) | set(df_com['Author'].unique()) - {np.nan}\n",
    "    \n",
    "    # Cleaning\n",
    "    df_posts = df_posts[~df_posts['Author'].isin(bots_and_deleted) & df_posts['Author'].notna()]\n",
    "    df_com = df_com[~df_com['Author'].isin(bots_and_deleted) & df_com['Author'].notna()]\n",
    "    \n",
    "    # Statistics after cleaning\n",
    "    posts_after = len(df_posts)\n",
    "    com_after = len(df_com)\n",
    "    unique_authors_after = set(df_posts['Author'].unique()) | set(df_com['Author'].unique())\n",
    "    \n",
    "    deleted_authors = unique_authors_before - unique_authors_after\n",
    "    \n",
    "    print(f\"--- Data Preparation Summary ---\")\n",
    "    print(f\"Unique authors removed: {len(deleted_authors)} ({', '.join(map(str, deleted_authors))})\")\n",
    "    print(f\"Posts removed: {posts_before - posts_after}\")\n",
    "    print(f\"Comments removed: {com_before - com_after}\")\n",
    "    print(f\"Total rows dropped: {(posts_before - posts_after) + (com_before - com_after)}\")\n",
    "    print(f\"--------------------------------\")\n",
    "\n",
    "    # Convert dates and get ISO week\n",
    "    print(\"Extracting time windows...\")\n",
    "    df_posts['Date & Time'] = pd.to_datetime(df_posts['Date & Time'])\n",
    "    df_com['Date & Time'] = pd.to_datetime(df_com['Date & Time'])\n",
    "    \n",
    "    df_posts['ISOYearWeek'] = df_posts['Date & Time'].dt.strftime('%G-W%V')\n",
    "    df_com['ISOYearWeek'] = df_com['Date & Time'].dt.strftime('%G-W%V')\n",
    "    \n",
    "    print(\"Building interactions...\")\n",
    "    # Edge type 1: Comment -> Post\n",
    "    df_com_post = df_com[df_com['Parent ID'].isna()].copy()\n",
    "    df_com_post = df_com_post.merge(df_posts[['Post ID', 'Author', 'Subreddit']], on='Post ID', how='inner', suffixes=('_com', '_post'))\n",
    "    df_com_post = df_com_post[df_com_post['Author_com'] != df_com_post['Author_post']]\n",
    "    \n",
    "    # Weight: 2 for replying to post, + 0.1 for every 10 chars of content\n",
    "    df_com_post['Weight'] = 2.0 + (df_com_post['Content'].str.len().fillna(0) / 100.0)\n",
    "    \n",
    "    edges_post = df_com_post[['Author_com', 'Author_post', 'Subreddit', 'ISOYearWeek', 'Weight']].rename(\n",
    "        columns={'Author_com': 'Source', 'Author_post': 'Target'}\n",
    "    )\n",
    "    \n",
    "    # Edge type 2: Comment -> Comment\n",
    "    df_com_com = df_com[df_com['Parent ID'].notna()].copy()\n",
    "    df_com_com = df_com_com.merge(df_com[['Comment ID', 'Author']], left_on='Parent ID', right_on='Comment ID', suffixes=('_com', '_parent'))\n",
    "    df_com_com = df_com_com[df_com_com['Author_com'] != df_com_com['Author_parent']]\n",
    "    \n",
    "    # Get Subreddit from Post\n",
    "    df_com_com = df_com_com.merge(df_posts[['Post ID', 'Subreddit']], on='Post ID', how='inner')\n",
    "    \n",
    "    # Weight: 1 for replying to comment, + 0.1 for every 10 chars\n",
    "    df_com_com['Weight'] = 1.0 + (df_com_com['Content'].str.len().fillna(0) / 100.0)\n",
    "    \n",
    "    edges_com = df_com_com[['Author_com', 'Author_parent', 'Subreddit', 'ISOYearWeek', 'Weight']].rename(\n",
    "        columns={'Author_com': 'Source', 'Author_parent': 'Target'}\n",
    "    )\n",
    "    \n",
    "    # Consolidate all edges\n",
    "    all_edges = pd.concat([edges_post, edges_com], ignore_index=True)\n",
    "    \n",
    "    # Aggregate edge weights for same Source->Target within the same Subreddit and Week\n",
    "    all_edges = all_edges.groupby(['Source', 'Target', 'Subreddit', 'ISOYearWeek'], as_index=False)['Weight'].sum()\n",
    "    \n",
    "    print(\"Saving consolidated edges...\")\n",
    "    if not os.path.exists('graphs'):\n",
    "        os.makedirs('graphs')\n",
    "        \n",
    "    all_edges.to_csv('graphs/all_edges_consolidated.csv', index=False)\n",
    "    \n",
    "    print(all_edges.head())\n",
    "    print(\"Pipeline completed.\")\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    clean_and_prepare_data()\n"
]

# Update the source of the first cell (index 0)
nb.cells[0].source = "".join(new_source)

# Write the notebook back
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook first cell updated successfully.")
