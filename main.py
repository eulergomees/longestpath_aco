import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

df = pd.read_csv('grafo1.csv') #Ler o CSV

G = nx.DiGraph()
G = nx.from_pandas_edgelist(df, source='origem', target='destino', edge_attr='custo')

pos = nx.spring_layout(G)
weights = list(nx.get_edge_attributes(G, 'custo').values())

edge_labels = nx.get_edge_attributes(G, 'custo')

nx.draw_networkx_nodes(G, pos, node_size=300)
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_labels,
    font_size=6,
    horizontalalignment='center',
    verticalalignment='baseline',
)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)

plt.show()
