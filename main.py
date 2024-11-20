import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------------------------------------------------------------------------
base = 'A'

if base == 'A':
    grafo_csv = 'grafo1.csv'
    vinicial = 1
    vfinal = 12
if base == 'B':
    grafo_csv = 'grafo2.csv'
    vinicial = 1
    vfinal = 20
if base == 'C':
    grafo_csv = 'grafo3.csv'
    vinicial = 1
    vfinal = 100

num_formigas = 40  # Número de formigas
fer_inicial = 0.01  # Feromônio inicial
maxit = 100  # Número máximo de iterações
evaporacao = 0.07  # Taxa de evaporação
alpha = 1  # Peso para o feromônio
beta = 2  # Peso para a visibilidade (inverso do custo)
rho = 0.1  # Taxa de evaporação do feromônio


# Defina o nó inicial e final


# -----------------------------------------------------------------------------------------------
# Ler o CSV
def ler_csv(nome_arquivo):
    df = pd.read_csv(nome_arquivo)
    G = nx.Graph()  # Cria um grafo vazio
    G = nx.from_pandas_edgelist(df, source='origem', target='destino', edge_attr='custo')
    return G


# Plota o grafo
def plota_grafo(Grafo):
    pos = nx.spring_layout(Grafo)
    edge_labels = nx.get_edge_attributes(Grafo, 'custo')
    nx.draw_networkx_nodes(Grafo, pos, node_size=300)
    nx.draw_networkx_edge_labels(
        Grafo, pos, edge_labels=edge_labels,
        font_size=6,
        horizontalalignment='center',
        verticalalignment='baseline',
    )
    nx.draw_networkx_edges(Grafo, pos)
    nx.draw_networkx_labels(Grafo, pos)
    return plt.show()


# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
# Leitura do grafo
G = ler_csv(grafo_csv)
print(G)

# Plota o grafo
plota_grafo(G)
