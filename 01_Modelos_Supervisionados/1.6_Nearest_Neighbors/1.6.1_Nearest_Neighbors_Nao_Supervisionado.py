########## 1.6.1. Vizinhos mais próximos não supervisionados  ##########

    # NearestNeighbors implementa o aprendizado não supervisionado dos vizinhos mais próximos. Ele atua como uma interface uniforme para três algoritmos diferentes de vizinhos mais próximos: BallTree, KDTree e um algoritmo de força bruta baseado em rotinas em sklearn.metrics.pairwise. A escolha do algoritmo de busca de vizinhos é controlada através da palavra-chave 'algoritmo', que deve ser ['auto', 'ball_tree', 'kd_tree', 'brute']. Quando o valor padrão 'auto' é passado, o algoritmo tenta determinar a melhor abordagem a partir dos dados de treinamento. Para uma discussão sobre os pontos fortes e fracos de cada opção, consulte Algoritmos do vizinho mais próximo.


    # Aviso: Quanto aos algoritmos de Vizinhos mais próximos, se dois vizinhos k + 1 ek têm distâncias idênticas, mas rótulos diferentes, o resultado dependerá da ordem dos dados de treinamento. 


##### 1.6.1.1 Encontrar os vizinhos mais próximos 

    # Para a tarefa simples de encontrar os vizinhos mais próximos entre dois conjuntos de dados, os algoritmos não supervisionados em sklearn.neighbors podem ser usados: 

from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print(indices)
print(distances)

    # Como o conjunto de consultas corresponde ao conjunto de treinamento, o vizinho mais próximo de cada ponto é o próprio ponto, a uma distância zero.

    # Também é possível produzir de forma eficiente um gráfico esparso mostrando as conexões entre pontos vizinhos: 

print(nbrs.kneighbors_graph(X).toarray())

    # O conjunto de dados é estruturado de forma que os pontos próximos na ordem do índice estejam próximos no espaço de parâmetros, levando a uma matriz de bloco diagonal de K vizinhos mais próximos. Esse gráfico esparso é útil em uma variedade de circunstâncias que fazem uso de relações espaciais entre pontos para aprendizagem não supervisionada: em particular, consulte Isomap, LocallyLinearEmbedding e SpectralClustering. 


##### 1.6.1.2 Classes KDTree e BallTree 

    # Como alternativa, pode-se usar as classes KDTree ou BallTree diretamente para encontrar os vizinhos mais próximos. Esta é a funcionalidade envolvida pela classe NearestNeighbors usada acima. A Ball Tree e a KD Tree têm a mesma interface; vamos mostrar um exemplo de uso da árvore KD aqui:

from sklearn.neighbors import KDTree
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print(kdt.query(X, k=2, return_distance=False))




    # Consulte a documentação das classes KDTree e BallTree para obter mais informações sobre as opções disponíveis para pesquisas de vizinhos mais próximos, incluindo especificações de estratégias de consulta, métricas de distância, etc. Para uma lista de métricas disponíveis, consulte a documentação da classe DistanceMetric. 