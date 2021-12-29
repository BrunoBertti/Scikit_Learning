########## 2.2.1. Introdução ##########

    # Conjuntos de dados de alta dimensão podem ser muito difíceis de visualizar. Embora os dados em duas ou três dimensões possam ser plotados para mostrar a estrutura inerente dos dados, os gráficos de dimensões elevadas equivalentes são muito menos intuitivos. Para auxiliar na visualização da estrutura de um conjunto de dados, a dimensão deve ser reduzida de alguma forma.

    # A maneira mais simples de realizar essa redução de dimensionalidade é fazer uma projeção aleatória dos dados. Embora isso permita algum grau de visualização da estrutura de dados, a aleatoriedade da escolha deixa muito a desejar. Em uma projeção aleatória, é provável que a estrutura mais interessante dentro dos dados seja perdida. 

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

    # Para lidar com essa preocupação, uma série de estruturas de redução de dimensionalidade linear supervisionadas e não supervisionadas foram projetadas, como Análise de Componentes Principais (PCA), Análise de Componentes Independentes, Análise Discriminante Linear e outras. Esses algoritmos definem rubricas específicas para escolher uma projeção linear “interessante” dos dados. Esses métodos podem ser poderosos, mas geralmente perdem estruturas não lineares importantes nos dados. 


        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

    # O Manifold Learning pode ser pensado como uma tentativa de generalizar estruturas lineares como o PCA para serem sensíveis à estrutura não linear dos dados. Embora existam variantes supervisionadas, o problema típico de aprendizado múltiplo não é supervisionado: ele aprende a estrutura de alta dimensão dos dados a partir dos próprios dados, sem o uso de classificações predeterminadas. 



    ## Exemplos:

    ## See Manifold learning on handwritten digits: Locally Linear Embedding, Isomap… for an example of dimensionality reduction on handwritten digits. (https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py)

    ## See Comparison of Manifold Learning methods for an example of dimensionality reduction on a toy “S-curve” dataset. (https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py)


    # As múltiplas implementações de aprendizagem disponíveis no scikit-learn estão resumidas abaixo 

 