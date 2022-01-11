########## 3.2.2. Otimização de parâmetros aleatórios ##########

    # Embora o uso de uma grade de configurações de parâmetros seja atualmente o método mais usado para otimização de parâmetros, outros métodos de pesquisa têm propriedades mais favoráveis. RandomizedSearchCV implementa uma pesquisa aleatória sobre parâmetros, onde cada configuração é amostrada de uma distribuição sobre valores de parâmetros possíveis. Isso tem dois benefícios principais em relação a uma pesquisa exaustiva:

        # Um orçamento pode ser escolhido independentemente do número de parâmetros e valores possíveis.

        # Adicionar parâmetros que não influenciam o desempenho não diminui a eficiência. 

    
    # A especificação de como os parâmetros devem ser amostrados é feita usando um dicionário, muito semelhante à especificação de parâmetros para GridSearchCV. Além disso, um orçamento de computação, sendo o número de candidatos amostrados ou iterações amostrais, é especificado usando o parâmetro n_iter. Para cada parâmetro, pode ser especificada uma distribuição sobre valores possíveis ou uma lista de escolhas discretas (que serão amostradas uniformemente): 

{'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}


    # Este exemplo usa o módulo scipy.stats, que contém muitas distribuições úteis para parâmetros de amostragem, como expon, gamma, uniform ou randint.

    # Em princípio, pode ser passada qualquer função que forneça um método rvs (amostra variável aleatória) para amostrar um valor. Uma chamada para a função rvs deve fornecer amostras aleatórias independentes de possíveis valores de parâmetro em chamadas consecutivas. 


    # Atenção: As distribuições em scipy.stats anteriores à versão scipy 0.16 não permitem especificar um estado aleatório. Em vez disso, eles usam o estado aleatório numpy global, que pode ser propagado via np.random.seed ou definido usando np.random.set_state. No entanto, a partir do scikit-learn 0.18, o módulo sklearn.model_selection define o estado aleatório fornecido pelo usuário se scipy >= 0.16 também estiver disponível.
    # Para parâmetros contínuos, como C acima, é importante especificar uma distribuição contínua para aproveitar ao máximo a randomização. Dessa forma, aumentar n_iter sempre levará a uma pesquisa mais precisa.

    # Uma variável aleatória log-uniform contínua está disponível por meio de loguniform. Esta é uma versão contínua de parâmetros com espaçamento de log. Por exemplo, para especificar C acima, loguniform(1, 100) pode ser usado em vez de [1, 10, 100] ou np.logspace(0, 2, num=1000). Este é um alias para stats.reciprocal do SciPy.

    # Espelhando o exemplo acima na pesquisa de grade, podemos especificar uma variável aleatória contínua que é log uniformemente distribuída entre 1e0 e 1e3: 

from sklearn.utils.fixes import loguniform
{'C': loguniform(1e0, 1e3),
 'gamma': loguniform(1e-4, 1e-3),
 'kernel': ['rbf'],
 'class_weight':['balanced', None]}



    ## Exemplos:

    ## Comparing randomized search and grid search for hyperparameter estimation compares the usage and efficiency of randomized search and grid search. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py)


    ## Referências:

    ## Bergstra, J. and Bengio, Y., Random search for hyper-parameter optimization, The Journal of Machine Learning Research (2012)