import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import time
from multiprocessing import Pool, cpu_count


# Parametros
fer_inicial = 0.01
rho = 0.07  # Evaporação
maxit = 500
num_formigas = 50

# Escolher o CSV
base = 'D'


# Escolhe le o CSV e seleciona o vinicial e vfinal
def escolher_grafo(base):
    bases = {
        "A": ("data/exemplo_slides.csv", 1, 4),
        "B": ("data/grafo1.csv", 1, 12),
        "C": ("data/grafo2.csv", 1, 20),
        "D": ("data/grafo3.csv", 1, 100),
    }
    if base not in bases:
        raise ValueError('Não existe grafo!')

    grafo, vinicial, vfinal = bases[base]
    df = pd.read_csv(grafo)
    return df, vinicial, vfinal


# Plota grafo
def plota_grafo(grafo, melhor_caminho, vinicial, vfinal):
    pos = nx.spring_layout(grafo)
    edge_labels = nx.get_edge_attributes(grafo, 'weight')
    edge_colors = ['red' if (u, v) in melhor_caminho or (v, u) in melhor_caminho else 'black' for u, v in
                   grafo.edges()]
    nx.draw(
        grafo, pos, with_labels=True, edge_color=edge_colors, node_color='orange', font_weight='bold'
    )
    nx.draw_networkx_edge_labels(grafo, pos, edge_labels=edge_labels, verticalalignment='center', font_size=6)
    plt.show()


# Passa de CSV para grafo
def cria_grafo(info):
    grafo = nx.Graph()  # Cria uma grafo vazio
    origens, destinos, pesos = info.iloc[:, 0], info.iloc[:, 1], info.iloc[:, 2]
    for i in range(len(origens)):
        grafo.add_edge(origens[i], destinos[i], weight=pesos[i])  # Para cada no, adicione destino e peso
    return grafo, origens, destinos, pesos


# Inicializa matriz de feromonios
def inicializa_matriz_feromonios(num_arestas, fer_inicial):
    return np.full((num_arestas), fer_inicial)


# Atualizar os feromonios
def atualiza_feromonios(fer, caminhos, custos, origens, destinos, rho):
    fer += (1 - rho)
    for caminho, custo in zip(caminhos, custos):
        for i in range(len(caminho) - 1):
            origem, destino = caminho[i], caminho[i + 1]
            idx = origens[(origens == origem) & (destinos == destino)].index[0]
            fer[idx] += custo
            # Usar o inverso do custo para atualizar o feromonio = menor custo
            #fer[idx] += 1 / custo
    return fer


# Função para construir o caminho de uma formiga
def construir_caminho(args):
    origens, destinos, pesos, feromonios, vinicial, vfinal = args
    caminho, custo, vatual = [], 0, vinicial

    while vatual != vfinal:
        arestas_disponiveis = [
            (i, destinos[i]) for i in range(len(origens)) if origens[i] == vatual and destinos[i] not in caminho
        ]

        if not arestas_disponiveis:
            break

        probabilidades = np.array([pesos[i] * feromonios[i] for i, _ in arestas_disponiveis])
        probabilidades /= probabilidades.sum()

        escolha = np.random.choice(len(arestas_disponiveis), p=probabilidades)
        aresta_escolhida = arestas_disponiveis[escolha]

        caminho.append(aresta_escolhida[1])
        custo += pesos[aresta_escolhida[0]]
        vatual = aresta_escolhida[1]

    return caminho, custo


def aco(garfo, origens, destinos, pesos, vinicial, vfinal, maxit, num_formigas, fer_inicial, rho):
    melhor_caminho_global = None
    melhor_custo_global = 0
    feromonios = inicializa_matriz_feromonios(len(origens), fer_inicial)

    num_threads = cpu_count()  # Usar o número máximo de núcleos da CPU
    with Pool(processes=num_threads) as pool:
        for it in range(maxit):
            caminhos, custos = [], []

            # Paralelização usando multiprocessing
            resultados = pool.map(
                construir_caminho,
                [(origens, destinos, pesos, feromonios, vinicial, vfinal) for _ in range(num_formigas)]
            )

            for caminho, custo in resultados:
                if caminho:  # Ignorar caminhos inválidos
                    caminhos.append(caminho)
                    custos.append(custo)

            feromonios = atualiza_feromonios(feromonios, caminhos, custos, origens, destinos, rho)

            # Verificar melhor caminho da iteração (maior caminho)
            if custos:
                melhor_idx = np.argmax(custos)  # Maior caminho = argmax, Menor caminho = argmin
                if custos[melhor_idx] > melhor_custo_global:
                    melhor_caminho_global = caminhos[melhor_idx]
                    melhor_custo_global = custos[melhor_idx]

    return melhor_caminho_global, melhor_custo_global

# Execução
dados, vinicial, vfinal = escolher_grafo(base)
grafo, origens, destinos, pesos = cria_grafo(dados)

# Calculo de tempo
start_time = time.time()

melhor_caminho_global, melhor_custo_global = aco(
    grafo, origens, destinos, pesos, vinicial, vfinal, maxit, num_formigas, fer_inicial, rho
)

# Calcula o tempo total
end_time = time.time()
execution_time = end_time - start_time

# Exibe a melhor solução
print(f"Tempo de execução: {execution_time:.4f} segundos")
print(f"Melhor caminho: {', '.join(map(str, melhor_caminho_global))}")
print(f"Custo do melhor caminho: {melhor_custo_global}")

#Plota o grafo
melhor_caminho_edges = [(melhor_caminho_global[i], melhor_caminho_global[i + 1]) for i in range(len(melhor_caminho_global) - 1)]
plota_grafo(grafo, melhor_caminho_edges, vinicial, vfinal)