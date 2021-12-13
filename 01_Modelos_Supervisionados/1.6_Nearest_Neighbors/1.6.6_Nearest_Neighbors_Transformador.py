########## 1.6.6 Transformador de Vizinhos Mais Próximos  ##########

    # Muitos estimadores scikit-learn contam com os vizinhos mais próximos: vários classificadores e regressores, como KNeighborsClassifier e KNeighborsRegressor, mas também alguns métodos de clustering, como DBSCAN e SpectralClustering, e alguns embeddings múltiplos, como TSNE e Isomap.

    # Todos esses estimadores podem computar internamente os vizinhos mais próximos, mas a maioria deles também aceita o gráfico esparso dos vizinhos mais próximos pré-computado, conforme dado por kneighbours_graph e radius_neighbors_graph. Com modo mode = 'conectividade', essas funções retornam um grafo esparso de adjacência binário conforme necessário, por exemplo, em SpectralClustering. Enquanto com mode = 'distance', eles retornam um gráfico esparso de distância conforme necessário, por exemplo, no DBSCAN. Para incluir essas funções em um pipeline de scikit-learn, também é possível usar as classes correspondentes KNeighborsTransformer e RadiusNeighborsTransformer. Os benefícios desta API de gráfico esparso são múltiplos.

    # Primeiro, o gráfico pré-calculado pode ser reutilizado várias vezes, por exemplo, ao variar um parâmetro do estimador. Isso pode ser feito manualmente pelo usuário ou usando as propriedades de cache do pipeline scikit-learn: 

import tempfile
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression

cache_path = tempfile.gettempdir()    # usamos uma pasta temporária aqui 
X, _ = make_regression(n_samples=50, n_features=25, random_state=0)
estimator = make_pipeline(
    KNeighborsTransformer(mode='distance'),
    Isomap(n_components=3, metric='precomputed'),
    memory=cache_path)
X_embedded = estimator.fit_transform(X)
X_embedded.shape()


    # Em segundo lugar, o pré-cálculo do gráfico pode fornecer um controle mais preciso sobre a estimativa dos vizinhos mais próximos, por exemplo, permitindo o multiprocessamento através do parâmetro n_jobs, que pode não estar disponível em todos os estimadores.

    # Finalmente, a pré-computação pode ser realizada por estimadores customizados para usar diferentes implementações, como métodos de vizinhos mais próximos aproximados ou implementação com tipos de dados especiais. O gráfico esparso de vizinhos pré-computados precisa ser formatado como na saída radius_neighs_graph: 

        # uma matriz CSR (embora COO, CSC ou LIL sejam aceitos).

        # apenas armazene explicitamente os bairros mais próximos de cada amostra no que diz respeito aos dados de treinamento. Isso deve incluir aqueles a 0 distância de um ponto de consulta, incluindo a diagonal da matriz ao calcular as vizinhanças mais próximas entre os dados de treinamento e ele mesmo.

        # os dados de cada linha devem armazenar a distância em ordem crescente (opcional. Os dados não classificados serão classificados de forma estável, adicionando uma sobrecarga computacional).

        # todos os valores nos dados devem ser não negativos.

        # não deve haver índices duplicados em nenhuma linha (consulte https://github.com/scipy/scipy/issues/5807).

        # se o algoritmo que está sendo passado pela matriz pré-computada usa k vizinhos mais próximos (em oposição à vizinhança do raio), pelo menos k vizinhos devem ser armazenados em cada linha (ou k + 1, conforme explicado na nota a seguir). 

    # OBS: Quando um número específico de vizinhos é consultado (usando KNeighborsTransformer), a definição de n_neighbors é ambígua, pois pode incluir cada ponto de treinamento como seu próprio vizinho ou excluí-los. Nenhuma escolha é perfeita, pois incluí-los leva a um número diferente de vizinhos não próprios durante o treinamento e teste, enquanto excluí-los leva a uma diferença entre fit (X) .transform (X) e fit_transform (X), o que é contra o scikit -learn API. No KNeighborsTransformer, usamos a definição que inclui cada ponto de treinamento como seu próprio vizinho na contagem de n_neighs. No entanto, por razões de compatibilidade com outros estimadores que usam a outra definição, um vizinho extra será calculado quando modo == 'distância'. Para maximizar a compatibilidade com todos os estimadores, uma escolha segura é sempre incluir um vizinho extra em um estimador de vizinhos mais próximos personalizado, uma vez que vizinhos desnecessários serão filtrados pelos estimadores seguintes. 


    ## Exemplos:

    ## Approximate nearest neighbors in TSNE: an example of pipelining KNeighborsTransformer and TSNE. Also proposes two custom nearest neighbors estimators based on external packages. (https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html#sphx-glr-auto-examples-neighbors-approximate-nearest-neighbors-py)

    ## Caching nearest neighbors: an example of pipelining KNeighborsTransformer and KNeighborsClassifier to enable caching of the neighbors graph during a hyper-parameter grid-search. (https://scikit-learn.org/stable/auto_examples/neighbors/plot_caching_nearest_neighbors.html#sphx-glr-auto-examples-neighbors-plot-caching-nearest-neighbors-py)