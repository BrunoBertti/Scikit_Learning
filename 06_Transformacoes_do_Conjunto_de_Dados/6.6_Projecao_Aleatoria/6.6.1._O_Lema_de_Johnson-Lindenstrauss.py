########## 6.6.1. O lema de Johnson-Lindenstrauss ##########



    # O principal resultado teórico por trás da eficiência da projeção aleatória é o lema de Johnson-Lindenstrauss (citando a Wikipedia):

    # Em matemática, o lema de Johnson-Lindenstrauss é um resultado sobre embeddings de baixa distorção de pontos de alta dimensão em espaço euclidiano de baixa dimensão. O lema afirma que um pequeno conjunto de pontos em um espaço de alta dimensão pode ser incorporado em um espaço de dimensão muito menor de tal forma que as distâncias entre os pontos sejam praticamente preservadas. O mapa usado para a incorporação é pelo menos Lipschitz, e pode até ser considerado uma projeção ortogonal.

    # Conhecendo apenas o número de amostras, o johnson_lindenstrauss_min_dim estima conservadoramente o tamanho mínimo do subespaço aleatório para garantir uma distorção limitada introduzida pela projeção aleatória: 


from sklearn.random_projection import johnson_lindenstrauss_min_dim
johnson_lindenstrauss_min_dim(n_samples=1e6, eps=0.5)

johnson_lindenstrauss_min_dim(n_samples=1e6, eps=[0.5, 0.1, 0.01])

johnson_lindenstrauss_min_dim(n_samples=[1e4, 1e5, 1e6], eps=0.1)



        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_johnson_lindenstrauss_bound.html 



    ## Exemplos:

    ## See The Johnson-Lindenstrauss bound for embedding with random projections for a theoretical explication on the Johnson-Lindenstrauss lemma and an empirical validation using sparse random matrices. (https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_johnson_lindenstrauss_bound.html#sphx-glr-auto-examples-miscellaneous-plot-johnson-lindenstrauss-bound-py)






    ## Referências:

    ## Sanjoy Dasgupta and Anupam Gupta, 1999. An elementary proof of the Johnson-Lindenstrauss Lemma. ( http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.39.3334&rep=rep1&type=pdf)