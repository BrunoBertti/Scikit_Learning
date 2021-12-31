########## 2.3.1. Visão geral dos métodos de agrupamento ##########

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html



###### TABELA DE Nome: do método, Parâmetros, Escalabilidade, Caso de uso, Geometria (métrica usada).




    # O agrupamento de geometria não plana é útil quando os clusters têm uma forma específica, ou seja, uma variedade não plana, e a distância euclidiana padrão não é a métrica correta. Este caso surge nas duas linhas superiores da figura acima.

    # Modelos de mistura gaussiana, úteis para agrupamento, são descritos em outro capítulo da documentação dedicado a modelos de mistura. KMeans pode ser visto como um caso especial de modelo de mistura gaussiana com covariância igual por componente.

    # Os métodos de agrupamento transdutivo (em contraste com os métodos de agrupamento indutivo) não são projetados para serem aplicados a dados novos e invisíveis. 