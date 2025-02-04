import spacy
import pandas as pd
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

def visualize_graph(g):
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
            # pos[v] = (1.0, float(int(v)))   # Coloca termos à direita

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
        output="text_graph_center.pdf"
    )

def min_sbm(g, color_prop, size_prop, label_prop):
    #Inferindo comunidades usando o SBM de maneira mais simples possível
    state = minimize_blockmodel_dl(g)

    # Desenhar as comunidades inferidas com as per'sonalizações
    state.draw(
        vertex_fill_color=color_prop,   # Define a cor dos vértices
        vertex_size=size_prop,          # Define o tamanho dos vértices
        vertex_text=label_prop,         # Define o rótulo dos vértices (ID)
        vertex_font_size=8,             # Tamanho da fonte dos rótulos
        output_size=(800, 800),         # Tamanho da saída
        output="text_graph_sbm.pdf"     # Arquivo PDF de saída
    )

def edge_matrix():
    # Reorganizar os nós para garantir que estejam em ordem contígua
    b = contiguous_map(state.get_blocks())  # Use contiguous_map diretamente
    state = state.copy(b=b)  # Cria uma cópia do estado com blocos reorganizados

    # Obter a matriz de contagem de arestas entre blocos
    e = state.get_matrix()  # Obtém a matriz esparsa de conectividade entre blocos

    # Número de blocos não vazios
    B = state.get_nonempty_B()  # Retorna o número de blocos que contêm vértices

    # Visualizar a matriz de contagem de arestas
    from matplotlib.pyplot import matshow, savefig
    matshow(e.todense()[:B, :B])  # Converte para matriz densa e visualiza os blocos não vazios
    savefig("text-edge-counts.svg")  # Salva a matriz visualizada como SVG

def nested_sbm(g):
    # Nested SBM
    state = minimize_nested_blockmodel_dl(g)
    state.draw(bg_color='white',
        output="text-hsbm-fit.svg"
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
    print(f"Nível 0: O nó 33 pertence ao bloco {r}")

    # Nível 1: Bloco ao qual o bloco de nível 0 foi agrupado
    r = levels[1].get_blocks()[r]  # Atribuição do bloco de nível superior para o bloco do nível 0
    print(f"Nível 1: O bloco anterior foi agrupado no bloco {r}")

    # Nível 2: Bloco ao qual o bloco de nível 1 foi agrupado
    r = levels[2].get_blocks()[r]  # Atribuição do bloco de nível superior para o bloco do nível 1
    print(f"Nível 2: O bloco anterior foi agrupado no bloco {r}")

    # Discussão sobre o que esperar:
    # - Nível 0: A partição mais detalhada, com o maior número de blocos. 
    #   Espera-se que os blocos representem comunidades finas baseadas em conexões locais.
    # - Nível 1: Os blocos de nível 0 foram agrupados em comunidades maiores. 
    #   Este nível pode capturar conexões entre comunidades vizinhas.
    # - Nível 2 (e níveis superiores): Os blocos continuam sendo agrupados, eventualmente
    #   formando uma única comunidade no nível mais alto. Esse nível reflete a visão mais geral
    #   do grafo como um todo.

def main():
    start_time = time.time()
    # Carregar spaCy
    nlp = spacy.load("en_core_web_sm")
    # Carregar o DataFrame
    df = pd.read_parquet("wos_sts_journals.parquet")
    # print(df["abstract"].map(len))
    g = initialize_graph()
    # # Selecionar uma amostra do DataFrame
    df = df.sample(n=200, random_state=42)
    
    g = build_graph(g,df,nlp)
    visualize_graph(g)
    # min_sbm(g)
    # edge_matrix()
    # nested_sbm(g)
    print(f"\n\nO tempo total de execução desse código foi de :{time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()