import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


def load_embeddings(filepath):
    """
    Carrega embeddings de um arquivo.
    Formato: primeira linha contém N (num nós) e K (dimensões)
    Linhas seguintes: nome_nó valor1 valor2 ... valorK
    """
    embeddings = {}
    node_names = []
    
    with open(filepath, 'r') as f:
        # Primeira linha: N K
        first_line = f.readline().strip().split()
        n_nodes = int(first_line[0])
        n_dims = int(first_line[1])
        
        # Ler embeddings
        embedding_matrix = []
        for line in f:
            parts = line.strip().split()
            node_name = parts[0]
            embedding = list(map(float, parts[1:]))
            
            node_names.append(node_name)
            embedding_matrix.append(embedding)
            embeddings[node_name] = embedding
    
    return embeddings, np.array(embedding_matrix), node_names, n_dims


def load_clustering(filepath):
    """
    Carrega resultados de clusterização.
    Formato: node_id cluster_label
    """
    clusters = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_name = parts[0]
            cluster_label = int(parts[1])
            clusters[node_name] = cluster_label
    
    return clusters


def load_graph(filepath):
    """
    Carrega grafo a partir de um arquivo de arestas.
    Suporta edgelist, .mtx, .cites e arquivos de pares simples.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.mtx':
        from scipy.io import mmread

        try:
            matrix = mmread(filepath)
            if hasattr(matrix, 'tocoo'):
                matrix = matrix.tocoo()

            G = nx.Graph()
            for i, j in zip(matrix.row, matrix.col):
                if i == j:
                    continue
                G.add_edge(int(i), int(j))
            return G
        except Exception:
            G = nx.Graph()
            edge_pairs = []

            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('%') or stripped.startswith('#'):
                        continue

                    parts = stripped.split()
                    if len(parts) < 2:
                        continue

                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                    except ValueError:
                        continue

                    edge_pairs.append((u, v))

            if not edge_pairs:
                raise ValueError(f'Nenhum par de arestas encontrado em {filepath}')

            if len(edge_pairs) > 1:
                first_u, first_v = edge_pairs[0]
                other_nodes = [max(u, v) for u, v in edge_pairs[1:]]
                if other_nodes and first_u >= max(other_nodes) and first_v >= max(other_nodes):
                    edge_pairs = edge_pairs[1:]

            for u, v in edge_pairs:
                if u == v:
                    continue
                G.add_edge(u, v)

            if G.number_of_nodes() == 0:
                raise ValueError(f'Nenhum nó encontrado em {filepath}')
            return G

    if ext in {'.cites', '.txt', '.csv', '.edges', '.edge', '.dat'}:
        G = nx.Graph()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue

                parts = stripped.split()
                if len(parts) < 2:
                    continue

                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except ValueError:
                    continue

                G.add_edge(u, v)

        if G.number_of_nodes() == 0:
            raise ValueError(f'Nenhum nó encontrado em {filepath}')
        return G

    return nx.read_edgelist(filepath, nodetype=int)


def find_graph_file(base_name, base_dir):
    """
    Procura um arquivo de arestas no diretório de datasets.
    """
    dataset_dir = os.path.join(base_dir, 'datasets')
    graph_dir = os.path.join(base_dir, 'graph')

    if os.path.exists(dataset_dir):
        for root, _, files in os.walk(dataset_dir):
            for filename in files:
                name, ext = os.path.splitext(filename)
                if name != base_name and not filename.startswith(base_name):
                    continue

                if ext.lower() in {'.mtx', '.cites', '.txt', '.csv', '.edges', '.edge', '.edgelist', '.dat'}:
                    return os.path.join(root, filename)

    if os.path.exists(graph_dir):
        for filename in os.listdir(graph_dir):
            if filename.startswith(base_name) and filename.endswith(('.edgelist', '.txt', '.csv', '.edges', '.edge', '.dat')):
                return os.path.join(graph_dir, filename)

    return None


def get_cluster_colors(clusters):
    """
    Cria um mapa de cores para os clusters.
    """
    unique_clusters = sorted(set(clusters.values()))
    n_clusters = len(unique_clusters)
    
    # Usar colormap para gerar cores
    cmap = plt.get_cmap('tab10' if n_clusters <= 10 else 'tab20')
    color_map = {cluster: cmap(i / n_clusters) for i, cluster in enumerate(unique_clusters)}
    
    return color_map


def plot_embedding_2d(embedding_matrix, node_names, clusters, output_filepath):
    """
    Faz plot do embedding em 2D com cores por cluster.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Obter mapa de cores
    color_map = get_cluster_colors(clusters)
    
    # Plotar pontos com cores por cluster
    for node_name, embedding in zip(node_names, embedding_matrix):
        cluster = clusters[node_name]
        color = color_map[cluster]
        ax.scatter(embedding[0], embedding[1], c=[color], s=100, alpha=0.7, edgecolors='black')
    
    # Adicionar legenda
    unique_clusters = sorted(set(clusters.values()))
    patches = [mpatches.Patch(color=color_map[cluster], label=f'Cluster {cluster}') 
               for cluster in unique_clusters]
    ax.legend(handles=patches, loc='best')
    
    ax.set_xlabel('Dimensão 1')
    ax.set_ylabel('Dimensão 2')
    ax.set_title('Embeddings 2D com Clusters')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Plot de embedding salvo: {output_filepath}")
    plt.close()


def plot_graph(G, clusters, output_filepath):
    """
    Faz plot do grafo com cores por cluster.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Obter mapa de cores
    color_map = get_cluster_colors(clusters)

    # Criar um grafo auxiliar com pesos para aproximar nós do mesmo cluster
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from(G.nodes())

    for u, v, data in G.edges(data=True):
        cluster_graph.add_edge(u, v, weight=data.get('weight', 1))

    node_to_cluster = {node: clusters[str(node)] for node in G.nodes()}
    for u, v in G.edges():
        if node_to_cluster[u] == node_to_cluster[v]:
            cluster_graph[u][v]['weight'] += 2.0

    pos = nx.spring_layout(
        cluster_graph,
        k=0.6,
        iterations=120,
        seed=42,
        weight='weight'
    )
    
    # Criar lista de cores para cada nó
    node_colors = [color_map[clusters[str(node)]] for node in G.nodes()]
    
    # Desenhar o grafo
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, 
                          alpha=0.8, ax=ax, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    # Adicionar legenda
    unique_clusters = sorted(set(clusters.values()))
    patches = [mpatches.Patch(color=color_map[cluster], label=f'Cluster {cluster}') 
               for cluster in unique_clusters]
    ax.legend(handles=patches, loc='best')
    
    ax.set_title('Grafo com Clusters')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Plot do grafo salvo: {output_filepath}")
    plt.close()


def main():
    # Caminhos
    base_dir = os.path.dirname(__file__)
    emb_dir = os.path.join(base_dir, 'emb')
    graph_dir = os.path.join(base_dir, 'graph')
    
    # Procurar por arquivos de embedding e clustering
    if not os.path.exists(emb_dir):
        print(f"Pasta '{emb_dir}' não encontrada!")
        return
    
    emb_files = [f for f in os.listdir(emb_dir) if f.endswith('.emd')]
    
    if not emb_files:
        print(f"Nenhum arquivo de embedding encontrado em '{emb_dir}'")
        return
    
    # Processar cada arquivo de embedding
    for emb_file in emb_files:
        emb_path = os.path.join(emb_dir, emb_file)
        base_name = os.path.splitext(emb_file)[0]
        clusters_file = f"{base_name}_clusters.txt"
        clusters_path = os.path.join(emb_dir, clusters_file)
        
        # Verificar se existe arquivo de clusters
        if not os.path.exists(clusters_path):
            print(f"⚠ Arquivo de clusters não encontrado: {clusters_file}")
            print(f"  Execute clustering.py primeiro!")
            continue
        
        # Procurar arquivo de arestas correspondente no diretório de datasets
        edgelist_path = find_graph_file(base_name, base_dir)
        
        if not edgelist_path:
            print(f"⚠ Arquivo de arestas não encontrado para '{base_name}' em datasets/ ou graph/")
            continue
        
        print(f"\nProcessando {emb_file}...")
        
        try:
            # Carregar dados
            embeddings, embedding_matrix, node_names, n_dims = load_embeddings(emb_path)
            clusters = load_clustering(clusters_path)
            G = load_graph(edgelist_path)
            
            print(f"  - Nodes: {len(node_names)}")
            print(f"  - Dimensões: {n_dims}")
            print(f"  - Clusters: {len(set(clusters.values()))}")
            
            # Plot do embedding (se for 2D)
            if n_dims == 2:
                embedding_plot_path = os.path.join(emb_dir, f"{base_name}_embedding_2d.png")
                plot_embedding_2d(embedding_matrix, node_names, clusters, embedding_plot_path)
            else:
                print(f"  ⚠ Embedding não é 2D ({n_dims}D). Pulando plot de embedding.")
            
            # Plot do grafo
            graph_plot_path = os.path.join(emb_dir, f"{base_name}_graph.png")
            plot_graph(G, clusters, graph_plot_path)
            
        except Exception as e:
            print(f"✗ Erro ao processar {emb_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
