import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------------------------------------------------------------------------
# Parametros do ACO
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

alpha = 0.5  # Influência do feromônio
beta = 0.5  # influência da heurística
rho = 0.05  # Taxa de evaporação
Q = 10  # Feromonio depositada pelas formigas
t_inicial = 0.01  # Feromonio inicial
lim_iter = 500  # Maximo de iterações
num_formigas = 40 #numero de formigas


# -----------------------------------------------------------------------------------------------
# Grafo e leitura do .CSV
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
# Inicializa matriz de feromonios
def inicializa_matriz_feromonios(Grafo, feromonio_inicial):
    matriz_feromonios = {edge: feromonio_inicial for edge in Grafo.edges}
    return matriz_feromonios


# -----------------------------------------------------------------------------------------------
# ACO

# -----------------------------------------------------------------------------------------------
# Execução
G = ler_csv(grafo_csv)
print(inicializa_matriz_feromonios(G, t_inicial))

# Plota o grafo
plota_grafo(G)
