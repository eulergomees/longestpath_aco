import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------- VARIÁVEIS -------------------------------------
feromonio_inicial = 0.01
taxa_evaporacao = 0.07
maximo_iteracao = 200
quantidade_formigas = 80
quantidade_execucoes = 1
base_csv = "A"

# ------------------------- FUNÇÃO CSV ------------------------------------------
def carregar_base(base):
    if base == "A":
        arquivo_csv = pd.read_csv('data/grafo1.csv')
        vinicial, vfinal = 1, 12
        nome_base = "Grafo A - 12 vértices e 25 arestas"
    elif base == "B":
        arquivo_csv = pd.read_csv('data/grafo2.csv')
        vinicial, vfinal = 1, 20
        nome_base = "Grafo B - 20 vértices e 190 arestas"
    elif base == "C":
        arquivo_csv = pd.read_csv('data/grafo3.csv')
        vinicial, vfinal = 1, 100
        nome_base = "Grafo C - 100 vértices e 8020 arestas"
    else:
        raise ValueError("Base inválida.")
    return arquivo_csv, vinicial, vfinal, nome_base

# ----- DADOS CARREGADOS --------
arquivo_csv, vinicial, vfinal, nome_base = carregar_base(base_csv)
origens = arquivo_csv['origem'].values
destinos = arquivo_csv['destino'].values
pesos = arquivo_csv['custo'].values

# Transformar pesos para minimizar o caminho
pesos = 1 / pesos

# Criar grafo a partir das colunas de origem, destino e peso
grafo = nx.Graph()
for origem, destino, peso in zip(origens, destinos, pesos):
    grafo.add_edge(origem, destino, weight=peso)

# ----------------- Inicializações --------------------------------------------
feromonio = np.full(len(origens), feromonio_inicial)
melhor_caminho = None
custo_melhor_caminho = np.inf
custos_expcorrente = []

# ----------------- Execução --------------------------------------------------
for nexp in range(quantidade_execucoes):
    for iteracao in range(maximo_iteracao):
        caminhos = []
        custos = []

        for _ in range(quantidade_formigas):
            vertice_atual = vinicial
            caminho = [vertice_atual]
            custo = 0
            caminho_valido = True

            while vertice_atual != vfinal and caminho_valido:
                indices_vertice_atual = np.where((origens == vertice_atual) & ~(np.isin(destinos, caminho)))[0]
                if len(indices_vertice_atual) == 0:
                    caminho_valido = False
                    break

                probabilidades = feromonio[indices_vertice_atual] * pesos[indices_vertice_atual]
                if probabilidades.sum() == 0:
                    caminho_valido = False
                    break

                probabilidades /= probabilidades.sum()
                escolha = np.random.choice(indices_vertice_atual, p=probabilidades)

                vertice_atual = destinos[escolha]
                caminho.append(vertice_atual)
                custo += 1 / pesos[escolha]  # Voltar ao peso original

            if caminho_valido:
                caminhos.append(caminho)
                custos.append(custo)
            else:
                custos.append(np.inf)

        # Atualizar feromônio
        feromonio = (1 - taxa_evaporacao) * feromonio
        for idx, caminho in enumerate(caminhos):
            if custos[idx] < np.inf:
                for i in range(len(caminho) - 1):
                    origem, destino = caminho[i], caminho[i + 1]
                    indice = np.where((origens == origem) & (destinos == destino))[0][0]
                    feromonio[indice] += 1 / custos[idx]

        # Atualizar melhor caminho
        melhor_iteracao = np.argmin(custos)
        if custos[melhor_iteracao] < custo_melhor_caminho:
            custo_melhor_caminho = custos[melhor_iteracao]
            melhor_caminho = caminhos[melhor_iteracao]

        custos_expcorrente.append(custo_melhor_caminho)

# ----------------- Resultados Finais ------------------------------------------
print(f"Quantidade de execuções: {quantidade_execucoes}")
print(f"Melhor caminho: {', '.join(map(str, melhor_caminho))}")
print(f"Custo do melhor caminho: {custo_melhor_caminho}")
print(f"Custo médio: {np.mean([c for c in custos if c < np.inf])}")

# ----------------- Visualização -----------------------------------------------
# ----------------- Visualização -----------------------------------------------
plt.figure(figsize=(12, 8))

# Gerar posições dos nós
pos = nx.spring_layout(grafo, seed=42)  # Define um layout com posições fixas para visualização consistente

# Desenhar o grafo completo
nx.draw_networkx(grafo, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=700, font_size=10)

# Destacar o melhor caminho, se existir
if melhor_caminho:
    edges = [(melhor_caminho[i], melhor_caminho[i + 1]) for i in range(len(melhor_caminho) - 1)]
    nx.draw_networkx_edges(grafo, pos, edgelist=edges, edge_color="red", width=2)

# Título do grafo
plt.title(nome_base)
plt.show()


# Gráfico de Convergência
plt.figure(figsize=(10, 6))
plt.plot(range(len(custos_expcorrente)), custos_expcorrente, label="Custo do Melhor Caminho")
plt.xlabel("Iteração")
plt.ylabel("Custo do Melhor Caminho")
plt.title("Convergência do ACO")
plt.legend()
plt.grid()
plt.show()
