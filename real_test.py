import spacy
import pandas as pd
from graph_tool.all import *
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk


# Carregar o modelo spaCy
nlp = spacy.load("en_core_web_sm")

# Carregar o DataFrame
df = pd.read_parquet("wos_sts_journals.parquet")

# # Selecionar uma amostra
df_sample = df.sample(n=100, random_state=42)  # Ajuste para testar rapidamente

# Processar a coluna "abstract" para extrair entidades
def extract_entities(text):
    if pd.isna(text):
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Aplicar a extração de entidades à coluna "abstract"
df_sample["entities"] = df_sample["abstract"].apply(extract_entities)

# Criar o grafo
g = Graph(directed=False)
vertex_map = {}  # Mapeia entidades para vértices

# Propriedades para os nós e arestas
vertex_name = g.new_vertex_property("string")
edge_weight = g.new_edge_property("int")

# Adicionar entidades como nós e coocorrências como arestas
for entities in df_sample["entities"]:  # Itera sobre a lista de entidades extraídas de cada linha do DataFrame
    for i, (ent1, label1) in enumerate(entities):  # Itera sobre cada entidade (ent1) e seu rótulo na lista
        if ent1 not in vertex_map:  # Verifica se a entidade ent1 ainda não foi adicionada como nó
            v = g.add_vertex()  # Cria um novo vértice no grafo
            vertex_map[ent1] = v  # Mapeia o nome da entidade ao vértice criado
            vertex_name[v] = ent1  # Define o nome da entidade como propriedade do vértice

        for j, (ent2, label2) in enumerate(entities):  # Itera novamente sobre as entidades para criar pares (ent1, ent2)
            if i < j:  # Garante que cada par seja processado apenas uma vez (evita duplicação de arestas)
                if ent2 not in vertex_map:  # Verifica se a entidade ent2 ainda não foi adicionada como nó
                    v = g.add_vertex()  # Cria um novo vértice para a entidade ent2
                    vertex_map[ent2] = v  # Mapeia o nome da entidade ent2 ao vértice criado
                    vertex_name[v] = ent2  # Define o nome da entidade como propriedade do vértice

                # Procura uma aresta entre os nós das entidades ent1 e ent2
                e = g.edge(vertex_map[ent1], vertex_map[ent2], add_missing=True)
                if e is None:  # Se a aresta não existir, cria uma nova aresta
                    e = g.add_edge(vertex_map[ent1], vertex_map[ent2])
                
                edge_weight[e] += 1  # Incrementa o peso da aresta para contar coocorrências

# Associar as propriedades ao grafo
g.vertex_properties["name"] = vertex_name
g.edge_properties["weight"] = edge_weight

from graph_tool.draw import graph_draw

# Desenhando o grafo em um arquivo PDF
graph_draw(
    g,
    vertex_text=g.vertex_properties["name"],  # Nome das entidades nos vértices
    vertex_font_size=10,
    output_size=(800, 800),
    output="text_graph.pdf"
)

#Inferindo comunidades usando o SBM de maneira mais simples possível
state = minimize_blockmodel_dl(g)

#Método para gerar um arquivo pdf para visualizar as comunidades geradas
state.draw(
    vertex_text=g.vertex_properties["name"],  # Nome das entidades nos vértices
    vertex_font_size=10,
    output="text_graph-sbm.pdf"
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

import numpy as np

state = minimize_nested_blockmodel_dl(g)

S1 = state.entropy()

for i in range(1000): # this should be sufficiently large
    state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

S2 = state.entropy()

print("Improvement:", S2 - S1)