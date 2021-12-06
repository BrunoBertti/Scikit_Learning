########## 1.1.8 LARS Lasso ##########

    # LassoLars é um modelo de laço implemetado usando o algoritmo LARS e, ao contrário da implementação baseada na descida das coordenadas, isso produz a solução exata, que é linear por partes em função da norma de seus coeficientes. 

from sklearn import linear_model
reg = linear_model.LassoLars(alpha=0.1, normalize=False)
print(reg.fit([[0,0], [1,1], [0,1]]))
print(reg.coef_)

    ## Exemplos: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html#sphx-glr-auto-examples-linear-model-plot-lasso-lars-py

    # O algoritmo Lars fornece o caminho completo dos coeficientes ao longo do parâmetro de regularização quase de graça, portanto, uma operação comum é recuperar o caminho com uma das funções lars_path ou lars_path_gram. 

##### 1.1.8.1 Formulação Matematica

    # O algoritmo é semelhante à regressão progressiva passo a passo, mas em vez de incluir recursos em cada etapa, os coeficientes estimados são aumentados em uma direção equiangular para as correlações de cada um com o resíduo. 

    # Em vez de fornecer um resultado vetorial, a solução LARS consiste em uma curva denotando a solução para cada valor da norma l1 do vetor de parâmetros. O caminho completo dos coeficientes é armazenado na matriz coef_path_ da forma (n_features, max_features + 1). A primeira coluna é sempre zero. 

    ## Referencias: Original Algorithm is detailed in the paper Least Angle Regression by Hastie et al. (https://www-stat.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)