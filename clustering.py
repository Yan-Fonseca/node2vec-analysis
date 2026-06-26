import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def parse_args():
    """
    Parses the clustering arguments.
    """
    parser = argparse.ArgumentParser(description="Run clustering on node embeddings.")
    parser.add_argument('--clusters', type=int, default=0,
                        help='Number of clusters for KMeans. If <= 0, infer using the existing heuristic.')
    return parser.parse_args()


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
    
    return embeddings, np.array(embedding_matrix), node_names


def perform_clustering(embedding_matrix, n_clusters=None):
    """
    Realiza clusterização K-means nos embeddings.
    """
    if n_clusters is None or n_clusters <= 0:
        # Usar sqrt(n) como heurística para número de clusters
        n_clusters = max(2, int(np.sqrt(len(embedding_matrix))) - 1)
    
    # Normalizar os embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embedding_matrix)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized_embeddings)
    
    return labels, kmeans


def save_clustering_results(node_names, labels, output_filepath):
    """
    Salva resultados da clusterização em um arquivo.
    """
    with open(output_filepath, 'w') as f:
        for node_name, cluster_label in zip(node_names, labels):
            f.write(f"{node_name} {cluster_label}\n")


def main():
    args = parse_args()

    # Caminho do arquivo de embeddings na pasta emb
    emb_dir = os.path.join(os.path.dirname(__file__), 'emb')
    
    # Listar arquivos na pasta emb
    if not os.path.exists(emb_dir):
        print(f"Pasta '{emb_dir}' não encontrada!")
        return
    
    emb_files = [f for f in os.listdir(emb_dir) if f.endswith('.emd') or f.endswith('.txt')]
    
    if not emb_files:
        print(f"Nenhum arquivo de embedding encontrado em '{emb_dir}'")
        return
    
    # Processar cada arquivo
    for emb_file in emb_files:
        emb_path = os.path.join(emb_dir, emb_file)
        print(f"Processando {emb_file}...")
        
        try:
            # Carregar embeddings
            embeddings, embedding_matrix, node_names = load_embeddings(emb_path)
            
            # Realizar clusterização
            labels, kmeans = perform_clustering(embedding_matrix, n_clusters=args.clusters)
            
            # Salvar resultados
            output_filename = f"{os.path.splitext(emb_file)[0]}_clusters.txt"
            output_path = os.path.join(emb_dir, output_filename)
            save_clustering_results(node_names, labels, output_path)
            
            print(f"✓ Clusterização concluída: {output_filename}")
            print(f"  - Nós: {len(node_names)}")
            print(f"  - Clusters: {len(np.unique(labels))}")
            
        except Exception as e:
            print(f"✗ Erro ao processar {emb_file}: {e}")


if __name__ == "__main__":
    main()
