import spacy
import pandas as pd
from graph_tool.all import *
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

# Carregar spaCy
nlp = spacy.load("en_core_web_sm")

# Carregar o DataFrame
df = pd.read_parquet("wos_sts_journals.parquet")
# print(df.columns)
# exit

# Criar o grafo
g = Graph(directed=False)

# Definir propriedades
name_prop = g.new_vertex_property("string")
tipo_prop = g.new_vertex_property("string")
edge_weight = g.new_edge_property("int")
g.vp["name"] = name_prop
g.vp["tipo"] = tipo_prop
g.ep["weight"] = edge_weight

# # Selecionar uma amostra do DataFrame
df = df.sample(n=20, random_state=42)

# Iterar pelas linhas do DataFrame e adicionar vértices para os documentos
for index, row in df.iterrows():
    v1 = g.add_vertex()
    g.vp["name"][v1] = row["title"]
    g.vp["tipo"][v1] = "Document"

    doc = nlp(row["abstract"])

    # print("o titulo ",index," = ",row["title"])
    # print("")
    # print("o resumo ", row["abstract"])

    # Iterar pelos termos no texto processado
    for termo in doc:
        if not termo.is_stop and not termo.is_punct:
            # Verificar se o termo já existe no grafo
            existing_vertices = [v for v in g.vertices() if g.vp["name"][v] == termo.text] # AUMENTAR A COMPLEXIDADE DO IF AUMENTA A EFICIÊNCIA DO CÓDIGO?(IF not document)  DUVIDA PARA LARGA ESCALA EU ENTENDO A COMPLEXIDADE COMO O(1) PARA IF
 
            if existing_vertices:
                v2 = existing_vertices[0]
            else:
                v2 = g.add_vertex()
                g.vp["name"][v2] = termo.text
                g.vp["tipo"][v2] = "Term"
# TODO: Futuramente aproveitar a classe da entidade para organizar informações e como peso para as relações na montagem das comunidades

            # Verificar se existe uma aresta entre os vértices v1 e v2
            if not g.edge(v1, v2):
                e = g.add_edge(v1, v2)
                edge_weight[e] = 1
            else:
                edge_weight[g.edge(v1, v2)] += 1

# # Verificar os vértices do tipo "Document"
# print(f"Total de vértices final: {g.num_vertices()}")
# print(f"Total de arestas final: {g.num_edges()}")

# print("\nVértices do tipo 'Document':")
# for v in g.vertices():
#     if g.vp["tipo"][v] == "Document":
#         print(f"ID: {int(v)}, Nome: {g.vp['name'][v]}")




from graph_tool.draw import sfdp_layout, graph_draw

# Gerar posições para os vértices usando um layout por força, onde vértices mais conectados tendem a ficar no centro
pos = sfdp_layout(g)
# pos = sfdp_layout(g, eweight=g.ep["weight"])

# Ajustar as cores e o tamanho dos vértices
color_prop = g.new_vertex_property("vector<double>")
size_prop = g.new_vertex_property("double")
label_prop = g.new_vertex_property("string")
# pos = g.new_vertex_property("vector<double>")

for v in g.vertices():
    if g.vp["tipo"][v] == "Document":
        color_prop[v] = [1.0, 0.0, 0.0, 1.0]  # Vermelho (RGBA)
        size_prop[v] = 20  # Tamanho maior para documentos
        # pos[v] = (-1.0, float(int(v)))  # Coloca documentos à esquerda
    else:
        color_prop[v] = [0.0, 0.0, 1.0, 1.0]  # Azul (RGBA)
        size_prop[v] = 10  # Tamanho menor para termos
        # pos[v] = (1.0, float(int(v)))   # Coloca termos à direita

    # Mostrar o ID do vértice como rótulo
    label_prop[v] = str(int(v))

# Desenhar o grafo
graph_draw(
    g,
    pos=pos,
    vertex_fill_color=color_prop,   # Define a cor dos vértices
    vertex_size=size_prop,          # Define o tamanho dos vértices
    vertex_text=label_prop,         # Define o rótulo dos vértices (ID)
    vertex_font_size=8,             # Tamanho da fonte dos rótulos
    output="text_graph_center.pdf"
)

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

from matplotlib.pyplot import matshow, savefig  # Importar as funções para visualização

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