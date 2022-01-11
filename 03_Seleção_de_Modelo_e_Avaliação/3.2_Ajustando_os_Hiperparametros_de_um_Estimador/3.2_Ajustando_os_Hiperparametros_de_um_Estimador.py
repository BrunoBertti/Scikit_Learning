########## 3.2. Ajustando os hiperparâmetros de um estimador ##########

    # Hiperparâmetros são parâmetros que não são aprendidos diretamente nos estimadores. No scikit-learn, eles são passados como argumentos para o construtor das classes estimadoras. Exemplos típicos incluem C, kernel e gamma para Support Vector Classifier, alpha para Lasso, etc.

    # É possível e recomendado pesquisar no espaço de hiperparâmetros o melhor escore de validação cruzada.

    # Qualquer parâmetro fornecido ao construir um estimador pode ser otimizado dessa maneira. Especificamente, para encontrar os nomes e valores atuais de todos os parâmetros para um determinado estimador, use: 

        # estimator.get_params()

    # Uma pesquisa consiste em:

        # um estimador (regressor ou classificador como sklearn.svm.SVC());

        # um espaço de parâmetros;

        # um método para pesquisa ou amostragem de candidatos;

        # um esquema de validação cruzada; e

        # uma função de pontuação.

    # Duas abordagens genéricas para pesquisa de parâmetros são fornecidas no scikit-learn: para determinados valores, GridSearchCV considera exaustivamente todas as combinações de parâmetros, enquanto RandomizedSearchCV pode amostrar um determinado número de candidatos de um espaço de parâmetros com uma distribuição especificada. Ambas as ferramentas têm halvings sucessivos HalvingGridSearchCV e HalvingRandomSearchCV, que podem ser muito mais rápidos em encontrar uma boa combinação de parâmetros.

    # Depois de descrever essas ferramentas, detalhamos as melhores práticas aplicáveis ​​a essas abordagens. Alguns modelos permitem estratégias de busca de parâmetros especializadas e eficientes, descritas em Alternativas à busca de parâmetros de força bruta.

    # Observe que é comum que um pequeno subconjunto desses parâmetros possa ter um grande impacto no desempenho preditivo ou computacional do modelo, enquanto outros podem ser deixados com seus valores padrão. Recomenda-se a leitura da docstring da classe do estimador para obter uma compreensão mais precisa de seu comportamento esperado, possivelmente lendo a referência anexa à literatura.    