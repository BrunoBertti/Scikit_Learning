########## 1.11. Ensemble Methods -  Métodos de Conjunto ##########

    # O objetivo dos métodos de ensemble é combinar as previsões de vários estimadores de base construídos com um determinado algoritmo de aprendizagem, a fim de melhorar a generalização / robustez sobre um único estimador.

    # Duas famílias de métodos de conjunto são geralmente distinguidas: 

        # Nos métodos de cálculo da média, o princípio orientador é construir vários estimadores de forma independente e, em seguida, calcular a média de suas previsões. Em média, o estimador combinado é geralmente melhor do que qualquer um do estimador de base única porque sua variância é reduzida. 

            ## Exemplo: 
            ## https://scikit-learn.org/stable/modules/ensemble.html#bagging
            ## https://scikit-learn.org/stable/modules/ensemble.html#forest


        # Em contraste, em métodos de boosting, estimadores de base são construídos sequencialmente e tenta-se reduzir o viés do estimador combinado. A motivação é combinar vários modelos fracos para produzir um conjunto poderoso. 

            ## Exemplo:
            ## https://scikit-learn.org/stable/modules/ensemble.html#adaboost
            ## https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
