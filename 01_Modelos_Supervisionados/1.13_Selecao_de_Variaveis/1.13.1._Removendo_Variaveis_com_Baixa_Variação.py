########## 1.13.1. Removendo variáveis com baixa variação ##########


    # VarianceThreshold é uma abordagem simples de linha de base para a seleção de recursos. Ele remove todos os recursos cuja variação não atende a algum limite. Por padrão, ele remove todos os recursos de variação zero, ou seja, recursos que têm o mesmo valor em todas as amostras.

    # Como exemplo, suponha que temos um conjunto de dados com recursos booleanos e queremos remover todos os recursos que são um ou zero (ativado ou desativado) em mais de 80% das amostras. Características booleanas são variáveis aleatórias de Bernoulli, e a variância de tais variáveis é dada por 

        # \mathrm{Var}[X] = p(1 - p)


    # portanto, podemos selecionar usando o limite .8 * (1 - .8): 

from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)

    # Como esperado, VarianceThreshold removeu a primeira coluna, que tem uma probabilidade p = 5/6> 0,8 de conter um zero. 