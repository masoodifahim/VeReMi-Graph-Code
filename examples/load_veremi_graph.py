from pathlib import Path
import json
import pandas as pd
import torch

DEMO_ROOT = Path(__file__).resolve().parents[1]
CATEGORY = 'medium'  # small, medium, large

node_df = pd.read_parquet(DEMO_ROOT / 'data' / CATEGORY / 'nodes.parquet')
edge_df = pd.read_parquet(DEMO_ROOT / 'data' / CATEGORY / 'edges.parquet')
snapshot_df = pd.read_parquet(DEMO_ROOT / 'data' / CATEGORY / 'snapshot.parquet')

print('TABULAR REPRESENTATION')
print('Nodes:', node_df.shape)
print('Edges:', edge_df.shape)
print('Snapshot rows:', snapshot_df.shape)

graph = torch.load(DEMO_ROOT / 'pyg' / f'{CATEGORY}.pt', weights_only=False)

print('\nPYTORCH GEOMETRIC REPRESENTATION')
print(graph)
print('x shape:', tuple(graph.x.shape))
print('edge_index shape:', tuple(graph.edge_index.shape))
print('edge_attr shape:', tuple(graph.edge_attr.shape))
print('multiclass labels:', tuple(graph.edge_y_multiclass.shape))
print('binary labels:', tuple(graph.edge_y_binary.shape))
print('graph context:', tuple(graph.graph_context.shape))

with open(DEMO_ROOT / 'schemas' / 'node_input_features.json', encoding='utf-8') as f:
    node_features = json.load(f)
with open(DEMO_ROOT / 'schemas' / 'edge_input_features.json', encoding='utf-8') as f:
    edge_features = json.load(f)
with open(DEMO_ROOT / 'schemas' / 'graph_context_features.json', encoding='utf-8') as f:
    graph_features = json.load(f)

print('\nFEATURE DIMENSIONS')
print('Node features:', len(node_features))
print('Edge features:', len(edge_features))
print('Graph context features:', len(graph_features))

print('\nFirst 5 node features for first node:')
for name, value in zip(node_features[:5], graph.x[0, :5].tolist()):
    print(f'{name}: {value}')

print('\nFirst 5 edge features for first edge:')
if graph.edge_attr.shape[0] > 0:
    for name, value in zip(edge_features[:5], graph.edge_attr[0, :5].tolist()):
        print(f'{name}: {value}')