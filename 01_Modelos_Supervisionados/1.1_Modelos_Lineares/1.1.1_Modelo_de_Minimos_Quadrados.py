
########## 1.1 Modelos Lineares ##########
    
    # A seguir, estão uns conjuntos de métodos destinados à regressão em que se epsera que o valor alvo seja uma combinação linear das variáveis. Na notação matematica, se y^ é a previsão do valor.
        
        # y^(w,x) = w0 + w1x1 + ... + wpxp
    
    # Em todo o módulo, nós designamos o vetor w = (w1, ... , wp) como coef_ e w0 como intercept_.

    # Para realiza a classificação com modelos lineares generalizados, consulte Regressão Logística (https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)



########## 1.1.1 Mínimos Quadrados Ordinários ##########
    
    # LinearRegression treina um modelo linear com coeficientes w = (w1, ..., wp) para minimizar a soma do quadrados dos erros entre os alvos observados no dataset, e os alvos previstos pela aproximação linear. Matematicamente isso resolve um problema da forma:

        # min ||Xw - y||22

    # LinearRegression terá suas matrizes de métodos de ajuste X, y e armazenará os coeficientes do modelo linear em seu membro coef_:

from scipy.sparse.linalg.interface import LinearOperator
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0,0], [1,1], [2,2]], [0,1,2])
print(reg.coef_)

    # As estimativas dos coeficientes dos Mínimos Quadrados Ordinários depende da independência das variáveis. Quando as variáveis são correlacionadas e as colunas da matrix tem uma dependência de aproximidade linear, o desing da matrix torna-se próxima do singular e, como resultado, a estimativa de mínimos quadrados torna-se altamente sensível a erros aleatórios no alvo observado, produzindo uma grande variância. Essa situação de multicolinearidade pode surgir, por exemplo, quando os dados são coletados sem um planejamento experimental.

    ## Exemplos: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

##### 1.1.1.1 Mínimos Quadrados Não Negativos 

    # É possível restringir todos os coeficientes a serem não negativos, o que pode ser útil quando eles represetam alguma quantidades físicas ou naturalmente não negativas (por exemplo, contagens de frequência ou preços de bens). LinearRegression aceita um parâmetro positivo booleano: quando definido como True Non-Negativa Least Squares são, então, aplicados.

    ## Exemplos: https://scikit-learn.org/stable/auto_examples/linear_model/plot_nnls.html#sphx-glr-auto-examples-linear-model-plot-nnls-py

##### 1.1.1.2 Complexidade de Mínimos Quadrados Ordinários

    # A solução dos mínimos quadrados é computada usando o valor singular da decomposição de X. Se X é a matrix de forma (n_amostras, n_variáveis) esse método tem um custe de O(n amostra*n^2variáveis), assumindo que namostra >= nvariáveis.