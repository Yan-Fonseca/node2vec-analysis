#!/bin/bash

echo "============================================"
echo "Node2Vec Pipeline com Visualização"
echo "============================================"

# Passo 1: Gerar embeddings
echo ""
echo "[1/3] Gerando embeddings..."
python3 src/main.py --input datasets/$1 --output emb/$2 --dimensions $3 --p $4 --q $5 --walk-length $6

if [ $? -ne 0 ]; then
    echo "✗ Erro ao gerar embeddings!"
    exit 1
fi

echo "✓ Embeddings gerados com sucesso!"

# Passo 2: Realizar clusterização
echo ""
echo "[2/3] Realizando clusterização..."
python3 clustering.py --clusters $7

if [ $? -ne 0 ]; then
    echo "✗ Erro ao realizar clusterização!"
    exit 1
fi

echo "✓ Clusterização realizada com sucesso!"

# Passo 3: Criar visualizações
echo ""
echo "[3/3] Criando plots..."
python3 plot_graph.py

if [ $? -ne 0 ]; then
    echo "✗ Erro ao criar plots!"
    exit 1
fi

echo ""
echo "============================================"
echo "✓ Pipeline concluído com sucesso!"
echo "============================================"
echo ""
echo "Arquivos gerados em emb/:"
echo "  - $2 (embeddings)"
echo "  - ${2%.*}_clusters.txt (clusters)"
echo "  - ${2%.*}_embedding_2d.png (plot embedding 2D)"
echo "  - ${2%.*}_graph.png (plot do grafo)"
echo ""