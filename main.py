import spacy
import pandas as pd
import numpy as np
from graph_tool.all import *
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk
import sys
from matplotlib.pyplot import matshow, savefig  # Importar as funções para visualização
from graph_tool.draw import sfdp_layout, graph_draw
from tqdm import tqdm
import time 

def initialize_graph():
    # Criar o grafo
    g = Graph(directed=False)

    # Definir propriedades
    name_prop = g.new_vertex_property("string")
    type_prop = g.new_vertex_property("string")
    edge_weight = g.new_edge_property("int")
    color_prop = g.new_vertex_property("vector<double>")
    size_prop = g.new_vertex_property("double")
    label_prop = g.new_vertex_property("string")
    g.vp["color"] = color_prop
    g.vp["size"] = size_prop
    g.vp["label"] = label_prop
    g.vp["name"] = name_prop
    g.vp["tipo"] = type_prop
    g.ep["weight"] = edge_weight

    return g

def build_block_graph(block_graph, state, g):

    # Visualizo o grafo de blocos antes das tratativas
    visualize_graph_bl(block_graph, "outputs/text_block_graph_original.pdf")
    
    # Definir propriedades block graph
    name_prop = block_graph.new_vertex_property("string")
    type_prop = block_graph.new_vertex_property("string")
    edge_weight = block_graph.new_edge_property("int")
    number_vertex = block_graph.new_vertex_property("int")
    color_prop = block_graph.new_vertex_property("vector<double>")
    size_prop = block_graph.new_vertex_property("double")
    label_prop = block_graph.new_vertex_property("string")
    vertex_shape = block_graph.new_vertex_property("string")
    block_graph.vp["shape"] = vertex_shape
    block_graph.vp["color"] = color_prop
    block_graph.vp["size"] = size_prop
    block_graph.vp["label"] = label_prop
    block_graph.vp["name"] = name_prop
    block_graph.vp["tipo"] = type_prop
    block_graph.ep["weight"] = edge_weight
    block_graph.vp["nvertex"] = number_vertex
    print(block_graph)  # Print só para confirmar a adição das propriedades

    # Obter a atribuição de blocos original antes da limpeza
    blocks = state.get_blocks().a
    block_sizes = np.bincount(blocks)

    # Criar um dicionário que associa cada bloco do grafo de blocos aos seus vértices no grafo original
    block_to_vertices = {}

    for i in range(len(block_sizes)):  # Iterar sobre os índices dos blocos
        block_vertices = [v for v in range(len(state.get_blocks().a)) if state.get_blocks().a[v] == i]  
        block_to_vertices[i] = block_vertices  # Salvar no dicionário

        terms = [g.vp["name"][g.vertex(v)] for v in block_vertices if g.vp["tipo"][g.vertex(v)] == "Term"]
        docs = sum(1 for v in block_vertices if g.vp["tipo"][g.vertex(v)] == "Document")

        if terms and docs:
            block_graph.vp["tipo"][i] = "Ambos"
        elif terms:
            block_graph.vp["tipo"][i] = "Term"
        elif docs:
            block_graph.vp["tipo"][i] = "Document"
        else:
            block_graph.vp["tipo"][i] = "Desconhecido"

        # Atualiza a name do bloco para mostrar alguns termos (caso seja um bloco de termos)
        if terms:
            block_graph.vp["name"][i] = ", ".join(terms[:3])  # Mostra até 3 termos na label


    # Remove vértices vazios do grafo de blocos
    to_remove = [v for v in block_graph.vertices() if v.out_degree() == 0 and v.in_degree() == 0]
    for v in reversed(to_remove):  # Remover de trás para frente evita problemas de indexação
        block_graph.remove_vertex(v, fast=False)
    visualize_graph(block_graph, "outputs/text_block_graph.pdf")

    vertices = list(block_graph.vertices())
    for v in tqdm(vertices, desc="Building Block Graph SBM", total=len(vertices)):
        if block_graph.vp["tipo"][v] == "Document":
            block_graph.vp["color"][v] = [1.0, 0.0, 0.0, 1.0]  # Vermelho (RGBA)
            block_graph.vp["size"][v] = 20  # Tamanho maior para documentos
            # block_graph.vp["shape"][v] = "circle"  # Termo: Quadrado ou Retângulo
        else:
            block_graph.vp["color"][v] = [0.0, 0.0, 1.0, 1.0]  # Azul (RGBA)
            # block_graph.vp["size"][v] = 10  # Tamanho menor para termos
            # block_graph.vp["shape"][v] = "square"  # Termo: Quadrado ou Retângulo

    # Aplicação do SBM ao grafo de blocos
    state_bg = minimize_blockmodel_dl(block_graph)
    pos = sfdp_layout(block_graph)
    state_bg.draw(
        pos=pos,
        vertex_fill_color=block_graph.vp["color"],   # Define a cor dos vértices
        vertex_size=block_graph.vp["size"],          # Define o tamanho dos vértices
        vertex_text=block_graph.vp["name"],         # Exibe os rótulos dos vértices
        vertex_font_size=10,             # Tamanho da fonte dos rótulos
        # vertex_shape=block_graph.vp["shape"],        # Define o formato do vértice
        output_size=(800, 800),         # Tamanho da saída
        output="outputs/text_block_graph_sbm.pdf"     # Arquivo PDF de saída
    )
    return block_to_vertices



def build_graph(g, df, nlp):
    map_termos = {}
    # Iterar pelas linhas do DataFrame e adicionar vértices para os documentos
    for index,row in tqdm(df.iterrows(),desc="DF Interation", total=len(df)):
        v1 = g.add_vertex()
        g.vp["name"][v1] = row["title"]
        g.vp["tipo"][v1] = "Document"
        map_termos[row["title"]] = v1

        doc = nlp(row["abstract"])

        # Iterar pelos termos no texto processado
        for termo in tqdm(doc,desc=f"Processing Doc {index + 1}", leave = False):
            if not termo.is_stop and not termo.is_punct:
                # Verificar se o termo já existe no grafo
                if termo.text in map_termos:
                    v2 = map_termos[termo.text]
                else:
                    v2 = g.add_vertex()                
                    g.vp['name'][v2] = termo.text
                    g.vp["tipo"][v2] = "Term"
                    map_termos[termo.text] = v2

                # Verificar se existe uma aresta entre os vértices v1 e v2
                if not g.edge(v1, v2):
                    e = g.add_edge(v1, v2)
                    g.ep["weight"][e] = 1
                else:
                    g.ep["weight"][g.edge(v1, v2)] += 1
    
    return g

def visualize_graph(g, graph_name):
    # Gerar posições para os vértices usando um layout por força, onde vértices mais conectados tendem a ficar no centro
    pos = sfdp_layout(g)

    vertices = list(g.vertices())

    for v in tqdm(vertices, desc="Building Graph", total=len(vertices)):
        if g.vp["tipo"][v] == "Document":
            g.vp["color"][v] = [1.0, 0.0, 0.0, 1.0]  # Vermelho (RGBA)
            g.vp["size"][v] = 20  # Tamanho maior para documentos
        else:
            g.vp["color"][v] = [0.0, 0.0, 1.0, 1.0]  # Azul (RGBA)
            g.vp["size"][v] = 10  # Tamanho menor para termos

        # Mostrar o ID do vértice como rótulo
        g.vp["label"][v] = str(int(v))

    # Desenhar o grafo
    graph_draw(
        g,
        pos=pos,
        vertex_fill_color=g.vp["color"],   # Define a cor dos vértices
        vertex_size=g.vp["size"],          # Define o tamanho dos vértices
        vertex_text=g.vp["label"],         # Define o rótulo dos vértices (ID)
        vertex_font_size=8,             # Tamanho da fonte dos rótulos
        output=graph_name
    )

def visualize_graph_bl(g, graph_name):
    # Gerar posições para os vértices usando um layout por força, onde vértices mais conectados tendem a ficar no centro
    pos = sfdp_layout(g)

    # Desenhar o grafo
    graph_draw(
        g,
        pos=pos,
        output=graph_name
    )


def min_sbm_wew(g):
    # # Se não for passado um argumento, mas o grafo tem pesos, definir state_args automaticamente
    # if state_args is None and "weight" in g.ep:
    #     state_args = {"eweight": g.ep["weight"]}
    
    # #Inferindo comunidades usando o SBM de maneira mais simples possível
    # state = minimize_blockmodel_dl(g, **(state_args or {}))
    state = minimize_blockmodel_dl(g, state_args={"eweight": g.ep["weight"]})


    # Desenhar as comunidades inferidas com as per'sonalizações
    state.draw(
        vertex_fill_color=g.vp["color"],   # Define a cor dos vértices
        vertex_size=g.vp["size"],          # Define o tamanho dos vértices
        vertex_text=g.vp["label"],         # Define o rótulo dos vértices (ID)
        vertex_font_size=8,             # Tamanho da fonte dos rótulos
        output_size=(800, 800),         # Tamanho da saída
        output="outputs/text_graph_sbm.pdf"     # Arquivo PDF de saída
    )

    return state


def edge_matrix(state, fig_name, g):
    # Reorganizar os nós para garantir que estejam em ordem contígua
    b = contiguous_map(state.get_blocks())  # Use contiguous_map diretamente
    state = state.copy(b=b)  # Cria uma cópia do estado com blocos reorganizados

    # Obter a matriz de contagem de arestas entre blocos
    e = state.get_matrix()  # Obtém a matriz esparsa de conectividade entre blocos

    # Número de blocos não vazios
    B = state.get_nonempty_B()  # Retorna o número de blocos que contêm vértices
    # print(f"O grafo possui {B} comunidades.")

    # Visualizar a matriz de contagem de arestas
    matshow(e.todense()[:B, :B])  # Converte para matriz densa e visualiza os blocos não vazios
    savefig(fig_name)  # Salva a matriz visualizada como SVG

    #Retorna quantos vértices tem em cada bloco
    block_sizes = np.bincount(state.get_blocks().a)
    # Imprimir o número de vértices em cada grupo e qual o tipo dos vertices que estão nele
    for i, size in enumerate(block_sizes):
        # Identificar quais tipos de vértices estão presentes no bloco
        block_vertices = [v for v in range(len(state.get_blocks().a)) if state.get_blocks().a[v] == i] # Cria uma lista com os índices dos vértices que pertencem ao bloco de índice 'i'
        terms = sum(1 for v in block_vertices if g.vp["tipo"][g.vertex(v)] == "Term")
        docs = sum(1 for v in block_vertices if g.vp["tipo"][g.vertex(v)] == "Document")
        
        if terms > 0 and docs > 0:
            group_type = "Ambos"
        elif terms > 0:
            group_type = "Termo"
        elif docs > 0:
            group_type = "Documento"
        else:
            group_type = "Desconhecido"

        print(f"O grupo {i} possui {size} vértices e é classificado como {group_type}.")


def nested_sbm(g):
    # Nested SBM
    state = minimize_nested_blockmodel_dl(g)
    state.draw(bg_color='white',
        output="outputs/text-nsbm-fit.svg"
        )

    state.print_summary()

    # Obter os níveis hierárquicos do SBM
    levels = state.get_levels()  # Retorna uma lista de estados, cada um representando um nível hierárquico
    for s in levels:
        print(s)  # Exibe informações detalhadas sobre o nível atual
        if s.get_N() == 1:  # Se o nível tiver apenas 1 bloco, a hierarquia chegou ao nível mais alto
            break  # Interrompe a iteração, pois não há mais subdivisões a explorar

    # Nível 0: Bloco ao qual o nó 33 pertence na partição mais detalhada
    r = levels[0].get_blocks()[33]  # `get_blocks()` retorna a atribuição de blocos para cada nó no nível 0
    print(f"\n\nNível 0: O nó 33 pertence ao bloco {r}")

    # Nível 1: Bloco ao qual o bloco de nível 0 foi agrupado
    r = levels[1].get_blocks()[r]  # Atribuição do bloco de nível superior para o bloco do nível 0
    print(f"Nível 1: O bloco anterior foi agrupado no bloco {r}")

    # Nível 2: Bloco ao qual o bloco de nível 1 foi agrupado
    r = levels[2].get_blocks()[r]  # Atribuição do bloco de nível superior para o bloco do nível 1
    print(f"Nível 2: O bloco anterior foi agrupado no bloco {r}\n")

    return state

def nested_sbm_wew(g, output_name):
    # Nested SBM
    state = minimize_nested_blockmodel_dl(g, state_args={"eweight": g.ep["weight"]})
    state.draw(bg_color='white',
        output=output_name
        )

    # Resumo da hierarquia inferida
    # Este método imprime informações sobre cada nível hierárquico, incluindo:
    # - Número de vértices (N)
    # - Número de blocos (B) (comunidades não vazias)
    # - Estatísticas da estrutura do grafo em cada nível
    state.print_summary()

    # Obter os níveis hierárquicos do SBM
    levels = state.get_levels()  # Retorna uma lista de estados, cada um representando um nível hierárquico
    for s in levels:
        print(s)  # Exibe informações detalhadas sobre o nível atual
        if s.get_N() == 1:  # Se o nível tiver apenas 1 bloco, a hierarquia chegou ao nível mais alto
            break  # Interrompe a iteração, pois não há mais subdivisões a explorar

    # Inspecionar a partição hierárquica
    # Neste exemplo, analisaremos a comunidade do nó 33 em diferentes níveis da hierarquia.

    # Nível 0: Bloco ao qual o nó 33 pertence na partição mais detalhada
    r = levels[0].get_blocks()[33]  # `get_blocks()` retorna a atribuição de blocos para cada nó no nível 0
    print(f"\n\nNível 0: O nó 33 pertence ao bloco {r}")

    # Nível 1: Bloco ao qual o bloco de nível 0 foi agrupado
    r = levels[1].get_blocks()[r]  # Atribuição do bloco de nível superior para o bloco do nível 0
    print(f"Nível 1: O bloco anterior foi agrupado no bloco {r}")

    # Nível 2: Bloco ao qual o bloco de nível 1 foi agrupado
    r = levels[2].get_blocks()[r]  # Atribuição do bloco de nível superior para o bloco do nível 1
    print(f"Nível 2: O bloco anterior foi agrupado no bloco {r}\n")

    # Discussão sobre o que esperar:
    # - Nível 0: A partição mais detalhada, com o maior número de blocos. 
    #   Espera-se que os blocos representem comunidades finas baseadas em conexões locais.
    # - Nível 1: Os blocos de nível 0 foram agrupados em comunidades maiores. 
    #   Este nível pode capturar conexões entre comunidades vizinhas.
    # - Nível 2 (e níveis superiores): Os blocos continuam sendo agrupados, eventualmente
    #   formando uma única comunidade no nível mais alto. Esse nível reflete a visão mais geral
    #   do grafo como um todo.
    return state

def refine_mcmc(state_nested, g):
    S1 = state_nested.entropy()

    for i in tqdm(range(1000), desc="Refining", total=1000): # Esse alcance deve ser suficientemente grande
        state_nested.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    S2 = state_nested.entropy()

    print("Improvement:", S2 - S1)

    # g.mcmc_anneal(state_nested, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))

def main():
    start_time = time.time()
    # Carregar spaCy
    nlp = spacy.load("en_core_web_sm")
    # Carregar o DataFrame
    df = pd.read_parquet("wos_sts_journals.parquet")
    g = initialize_graph()
    # Selecionar uma amostra do DataFrame    
    df = df.sample(n=30, random_state=42)
    #Construção do grafo
    g = build_graph(g,df,nlp)
    print(g)
    visualize_graph(g, "outputs/text_graph.pdf")
    #Aplicação do sbm com a propriedade de peso nas arestas
    state_wew = min_sbm_wew(g)
    print(f"O grafo original possui {state_wew.get_B()} blocos após o SBM, sendo que {state_wew.get_nonempty_B()} não estão vazios.")
    #Construção do grafo de blocos

    block_graph = state_wew.get_bg()

    build_block_graph(block_graph, state_wew, g)

    #Matriz de arestas entre os blocos do grafo original
    edge_matrix(state_wew, "outputs/text-edge-counts.svg", g)
    #Aplicação do Nested SBM para o grafo original
    state_nested_wew = nested_sbm_wew(g, "outputs/text-edge-counts-nsbm.svg")
    #Aplicação do Nested SBM para o grafo de blocos
    state_nested_block_graph = minimize_nested_blockmodel_dl(block_graph)
    state_nested_block_graph.draw(bg_color='white',
        output="outputs/text-edge-counts-nsbm-block_graph.svg"
        )
    #Refinamento do state do Nested SBM
    # refine_mcmc(state_nested_wew, g)

    print(f"\nO tempo total de execução desse código foi de :{time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()