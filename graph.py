import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ------------------------ VARIÁVEIS -----------------------------------
feromonio_inicial = 0.01
taxa_evaporacao = 0.07
maximo_iteracao = 200
quantidade_formigas = 80
quantidade_execucoes = 1
base_csv = "A"


# ------------------------- FUNÇÃO CSV ---------------------------------
def carregar_base(base):
    if base == "A":
        arquivo_csv = pd.read_csv("data/grafo1.csv")
        vinicial, vfinal = 1, 12
        nomebase = "Grafo A - 12 vértices e 25 arestas"
    elif base == "B":
        arquivo_csv = pd.read_csv("data/grafo2.csv")
        vinicial, vfinal = 1, 20
        nomebase = "Grafo B - 20 vértices e 190 arestas"
    elif base == "C":
        arquivo_csv = pd.read_csv("data/grafo3.csv")
        vinicial, vfinal = 1, 100
        nomebase = "Grafo C - 100 vértices e 8020 arestas"
    elif base == "D":
        arquivo_csv = pd.read_csv("data/exemplo_slides.csv")
        vinicial, vfinal = 1, 4
        nomebase = "Grafo D"
    else:
        raise ValueError("Base inválida")
    return arquivo_csv, vinicial, vfinal, nomebase


# Carregar dados
arquivo_csv, vinicial, vfinal, nomebase = carregar_base(base_csv)

# Separar as colunas em listas
origens = arquivo_csv["origem"].values
destinos = arquivo_csv["destino"].values
pesos = arquivo_csv["custo"].values

# Criar grafo com NetworkX
grafo = nx.Graph()
for origem, destino, peso in zip(origens, destinos, pesos):
    grafo.add_edge(origem, destino, weight=peso)

# --------------------- Algoritmo ACO -----------------------------------
todos_custos = np.zeros((quantidade_execucoes, quantidade_formigas))
custos_expcorrente = np.zeros(maximo_iteracao)

melhor_caminho = []
custo_melhor_caminho = -np.inf

for nexp in range(quantidade_execucoes):
    feromonio = np.full(len(origens), feromonio_inicial)
    caminhos = [[] for _ in range(quantidade_formigas)]
    custos = np.full(quantidade_formigas, -np.inf)

    num_threads = cpu_count()  # Usar o número máximo de núcleos da CPU
    with Pool(processes=num_threads) as pool:
        for iter in range(maximo_iteracao):
            for n in range(quantidade_formigas):
                vertice_atual = vinicial
                caminho = [vertice_atual]
                custo = 0
                caminho_valido = True

                while vertice_atual != vfinal and caminho_valido:
                    indices_vertice_atual = [
                        i for i, (origem, destino) in enumerate(zip(origens, destinos))
                        if origem == vertice_atual and destino not in caminho
                    ]

                    if not indices_vertice_atual:
                        caminho_valido = False
                        break

                    probabilidades = feromonio[indices_vertice_atual] * pesos[indices_vertice_atual]
                    soma_probabilidades = probabilidades.sum()

                    if soma_probabilidades > 0:
                        probabilidades /= soma_probabilidades
                    else:
                        caminho_valido = False
                        break

                    escolha = np.random.choice(indices_vertice_atual, p=probabilidades)
                    vertice_atual = destinos[escolha]
                    caminho.append(vertice_atual)
                    custo += pesos[escolha]

                if caminho_valido:
                    caminhos[n] = caminho
                    custos[n] = custo

            feromonio *= (1 - taxa_evaporacao)
            for n in range(quantidade_formigas):
                if custos[n] > -np.inf:
                    caminho_atual = caminhos[n]
                    for i in range(len(caminho_atual) - 1):
                        origem = caminho_atual[i]
                        destino = caminho_atual[i + 1]
                        indice = [
                            idx for idx, (o, d) in enumerate(zip(origens, destinos))
                            if o == origem and d == destino
                        ][0]
                        feromonio[indice] += custos[n]

            if max(custos) > custo_melhor_caminho:
                custo_melhor_caminho = max(custos)
                melhor_caminho = caminhos[np.argmax(custos)]

            custos_expcorrente[iter] = custo_melhor_caminho

    todos_custos[nexp, :] = custos

# ----------------- Resultados Finais -----------------------------------
print("\nResultados Finais")
print("Quantidade de execuções:", quantidade_execucoes)
print("Melhor caminho:", melhor_caminho)
print("Custo do melhor caminho:", custo_melhor_caminho)
print("Custo médio:", np.mean(todos_custos[todos_custos > -np.inf]))

# ------------------- Destacar Melhor Caminho no Grafo -------------------
plt.figure(figsize=(10, 7))

# Ajustar layout
pos = nx.spring_layout(grafo)

# Destacar arestas do melhor caminho
edges_melhor_caminho = [(melhor_caminho[i], melhor_caminho[i + 1]) for i in range(len(melhor_caminho) - 1)]
edge_colors = ["red" if edge in edges_melhor_caminho or edge[::-1] in edges_melhor_caminho else "gray" for edge in
               grafo.edges()]
edge_widths = [3 if edge in edges_melhor_caminho or edge[::-1] in edges_melhor_caminho else 1 for edge in grafo.edges()]

# Plotar o grafo
nx.draw(
    grafo,
    pos,
    with_labels=True,
    node_size=700,
    node_color="skyblue",
    font_size=10,
    edge_color=edge_colors,
    width=edge_widths
)
nx.draw_networkx_edge_labels(
    grafo,
    pos,
    edge_labels=nx.get_edge_attributes(grafo, "weight")
)
plt.title("Melhor Caminho Destacado no Grafo")
plt.show()

# Gráfico de convergência
plt.figure(figsize=(10, 5))
plt.plot(range(1, maximo_iteracao + 1), custos_expcorrente, label="Melhor Custo")
plt.title("Convergência do ACO")
plt.xlabel("Iteração")
plt.ylabel("Custo do Melhor Caminho")
plt.legend()
plt.grid(True)
plt.show()
