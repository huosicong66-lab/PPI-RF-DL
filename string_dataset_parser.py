import pandas as pd

def load_string_csv(path, score_threshold=700):
    df = pd.read_csv(path)

    df['protein1'] = df['protein1'].apply(lambda x: x.split('.')[-1])
    df['protein2'] = df['protein2'].apply(lambda x: x.split('.')[-1])
    df['label'] = (df['combined_score'] >= score_threshold).astype(int)
    pairs = list(zip(df['protein1'], df['protein2']))
    labels = df['label'].tolist()
    proteins = set(df['protein1']).union(set(df['protein2']))
    return pairs, labels, list(proteins)