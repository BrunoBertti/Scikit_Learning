########## 1.1.10 Regressão Bayesiana ##########

    # As técnicas de regressão bayesiana podem ser usadas para incluir parâmetros de regularização no procedimento de estimativa: o parâmetro de regularização não é definido em sentido rígido, mas ajustado aos dados disponíveis. 

    # Isso pode ser feito através da introdução de antecedentes não informativos sobre os hiperparâmetros do modelo. A regularização l2 usada na regressão e classificação de Ridge é equivalente a encontrar uma estimativa máxima a posteriori sob um Gaussiano anterior sobre os coeficientes w com precisão LAMBDA ^ -1. Em vez de definir lambda manualmente, é possível tratá-lo como uma variável aleatória para ser estimado a partir dos dados. 

    # Para obter um modelo totalmente probabilístico, a saída y é assumida como distribuída gaussiana em torno de Xw 

        # p(y|X, w, alpha) = N(y|Xw, alpha)

    # onde alfa é novamente tratado como uma variável aleatória que deve ser estimada a partir dos dados. 

    # As vantagens são:
        # Ele se adapta aos dados disponíveis.

        # Pode ser usado para incluir parâmetros de regularização no procedimento de estimativa. 

    # As desvantagens são:
        # A inferência pode ser demorada

    ## Referências:
    ## A good introduction to Bayesian methods is given in C. Bishop: Pattern Recognition and Machine learning
    ## Original Algorithm is detailed in the book Bayesian learning for neural networks by Radford M. Neal


##### 1.1.10.1 Regressão BAyesiana Ridge

    # BayesianRidge estima um modelo probabilístico do problema de regressão conforme descrito acima. O coefiente previo w é dado por um Gausiano Esferico.

        # p(w|lambda) = N(w|0, lambda^-1Ip)

    # As prévias sobre alfa e lambda são escolhidos para serem distribuições gama, o conjugado prévio para a precisão do Gaussiano. O modelo resultante é denominado Bayesian Ridge Regression e é semelhante ao Ridge clássico. 

    # Os parâmetros w, alfa e lambda são estimados conjuntamente durante o ajuste do modelo, os parâmetros de regularização alfa e lambda sendo estimados maximizando a probabilidade marginal logarítmica. A implementação do scikit-learn é baseada no algoritmo descrito no Apêndice A de (Tipping, 2001), onde a atualização dos parâmetros alpha e lambda é feita conforme sugerido em (MacKay, 1992). O valor inicial do procedimento de maximização pode ser definido com os hiperparâmetros alpha_init e lambda_init. 

    # Há mais 4 hiperparametros, alpha1, alpha2, lambda1 e lambda2 das distribuições anteriores gama sobre alpha e lambda. Geralmente são escolhidos para não serem informativos. Por padrão alhpa1 = alpha2 = lambda1 = lambda2 = 10^-6

    # Regressão Bayesiana Rigde é usado por regressão.

from sklearn import linear_model
X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y = [0., 1., 2., 3.]
reg = linear_model.BayesianRidge()
print(reg.fit(X,Y))

    # Depois de estar treinado, o modelo pode entao ser usado apra prever novos valores.

print(reg.predict([[1,0]]))

    # Os coeficientes w do modelo pode ser acessados.

print(reg.coef_)

    # Devido à estrutura bayesiana, os pesos encontrados são ligeiramente diferentes daqueles encontrados pelos Mínimos Quadrados Ordinários. No entanto, Bayesian Ridge Regression é mais robusto para problemas mal colocados. 

    ## Exemplos: 
    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge.html#sphx-glr-auto-examples-linear-model-plot-bayesian-ridge-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge_curvefit.html#sphx-glr-auto-examples-linear-model-plot-bayesian-ridge-curvefit-py

    ## Referências:
    ## Section 3.3 in Christopher M. Bishop: Pattern Recognition and Machine Learning, 2006

    ## David J. C. MacKay, Bayesian Interpolation, 1992. (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.9072&rep=rep1&type=pdf)

    ## Michael E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine, 2001. (http://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf)


##### 1.1.10.2 Determinação Automática de Relevância - ARD

    # ARDRegression é muito semelhante à Bayesian Ridge Regression, mas pode levar a coeficientes mais esparsos w. ADRRegression apresenta uma prévia diferente sobre w, eliminando a suposição de que o gaussiano é esférico. 

    # Em vez disso, a distribuição sobre w é considerada uma distribuição elíptica gaussiana paralela ao eixo. 

    # Isso significa que cada coeficiente wi é extraído de uma distribuição gaussiana, centrada em zero e com um lambdai de precisão

        # p(w|lambda) = N(w|0, A^-1)
    
    # com diag(A) = lambda = {lambda1, ..., lambdap}

    # Em contraste com a regressão bayesiana de Rigde, cada coordenada de wi tem seu próprio desvio padrão lambdai. O prior sobre todos os lambdai é escolhido para ser a mesma distribuição gama dada pelos hiperparâmetros lambda1 e lambda2 

    # ARD também é conhecido na literatura como Sparse Bayesian Learning and Relevance Vector Machine 

    ## Exemplos:
    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py

    ## Refenrências:

    ## Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 7.2.1

    ## David Wipf and Srikantan Nagarajan: A new view of automatic relevance determination (https://papers.nips.cc/paper/3372-a-new-view-of-automatic-relevance-determination.pdf)

    ## Michael E. Tipping: Sparse Bayesian Learning and the Relevance Vector Machine (http://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf)

    ## Tristan Fletcher: Relevance Vector Machines explained (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.651.8603&rep=rep1&type=pdf)
