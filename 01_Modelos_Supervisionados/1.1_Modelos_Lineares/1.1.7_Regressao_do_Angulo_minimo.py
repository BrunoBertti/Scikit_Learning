########## 1.1.7 Regressão do Ângulo Mínimo (LARs) ##########

    # Regressão do Ângulo Mínimo (LARS, em inglês) é um algoritmo de regressão para dados de alto-dimencionamento. desenvolvido por Bradley Efron, Trevor Hastie, Iain Johnstone e Robert Tibshirani. RAM é semelhante à regressão progressiva passo-a-passo. Em cada etapa, ele encontra a variável mais correlacionada com o alvo. Quando há muitas variáveis com igual correação, ao invés de continiar na mesma variável, ele segue uma direção equiangular entre as variáveis.

    # As vantangens do RAM são:
        # É numericamente eficiente em contextos onde o número de recursos é significativamente maior do que o número de amostras. 

        # É computacionalmente tão rápido quanto a seleção direta e tem a mesma ordem de complexidade dos mínimos quadrados comuns. 

        # Ele produz um caminho de solução linear completo por partes, que é útil na validação cruzada ou em tentativas semelhantes de ajustar o modelo. 

        # Se duas características são quase igualmente correlacionadas com o alvo, então seus coeficientes devem aumentar aproximadamente na mesma taxa. O algoritmo, portanto, se comporta como a intuição esperaria e também é mais estável.

        # É facilmente modificado para produzir soluções para outros estimadores, como o Lasso. 

    # As desvantagens são:

        # Como o LARS é baseado em um reajuste iterativo dos resíduos, ele parece ser especialmente sensível aos efeitos do ruído. Este problema é discutido em detalhes por Weisberg na seção de discussão do Efron et al. (2004) Artigo dos Anais de Estatística. 

    # O modelo LARS pode ser usado através do estimador Lars, ou sua implementação de baixo nível lars_path ou lars_path_gram. 