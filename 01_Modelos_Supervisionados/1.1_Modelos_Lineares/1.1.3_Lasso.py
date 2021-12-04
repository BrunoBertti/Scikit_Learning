########## 1.1.3 Lasso ##########
    # O Lasso é um modelo linear que estima coeficientes esparsos. É útil em alguns contextos devido à sua tendência de preferir soluções com menos coeficientes diferentes de zero, reduzindo efetivamente o número de recursos dos quais a solução dada é dependente. Por esta razão, Lasso e suas variantes são fundamentais para o campo da detecção por compressão. Sob certas condições, ele pode recuperar o conjunto exato de coeficientes diferentes de zero (consulte Detecção de compressão: reconstrução de tomografia com L1 anterior (Lasso)). (https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#sphx-glr-auto-examples-applications-plot-tomography-l1-reconstruction-py)

    # Matematicamente, consiste em um modelo linear com um termo de regularização adicionado. A função objetivo a minimizar é: 

        # min 1/2nsamples ||Xw - y||22 + alpha||w||1
    
    # A estimativa de laço, portanto, resolve a minimização da penalidade de mínimos quadrados com alpha||w||1 adicionado, onde alpha é uma constante e ||w||1 é a l1 norma do vetor de coeficientes. 

    # A implementação na classe Lasso usa a descida das coordenadas como o algoritmo para ajustar os coeficientes. Veja Regressão de Ângulo Mínimo para outra implementação:  (https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression)

from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
print(reg.fit([[0,0], [1,1]], [0,1]))
print(reg.predict([[1,1]]))

    # A função lasso_path é útil para tarefas de nível inferior, pois calcula os coeficientes ao longo do caminho completo de valores possíveis. 

    ## Exemplos:
    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py

    ## https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#sphx-glr-auto-examples-applications-plot-tomography-l1-reconstruction-py

    ## https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py

    # Notas: Seleção de Variáveis com Lasso
        # Como a regressão Lasso produz modelos esparsos, ela pode ser usada para realizar a seleção de recursos, conforme detalhado na seleção de recursos com base em L1. (https://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection)

    # As duas referências a seguir explicam as iterações usadas no solucionador de descida por coordenadas do scikit-learn, bem como o cálculo de lacuna de dualidade usado para controle de convergência. 

    # Referências:
        #  “Regularization Path For Generalized linear Models by Coordinate Descent”, Friedman, Hastie & Tibshirani, J Stat Softw, 2010 (Paper).
        
        #  “An Interior-Point Method for Large-Scale L1-Regularized Least Squares,” S. J. Kim, K. Koh, M. Lustig, S. Boyd and D. Gorinevsky, in IEEE Journal of Selected Topics in Signal Processing, 2007 (Paper)

##### 1.1.3.1 Definindo o paramatro de regularização

    # O parâmetro alfa controla o grau de dispersão dos coeficientes estimados. 

##### 1.1.3.1.1 Usando validação cruzada

    # scikit-learn expõe objetos que definem o parâmetro Lasso alpha por validação cruzada: LassoCV e LassoLarsCV. LassoLarsCV é baseado no algoritmo Least Angle Regression explicado abaixo.

    # Para conjuntos de dados de alta dimensão com muitos recursos colineares, LassoCV é mais frequentemente preferível. No entanto, o LassoLarsCV tem a vantagem de explorar os valores mais relevantes do parâmetro alfa e, se o número de amostras for muito pequeno em comparação com o número de recursos, geralmente é mais rápido do que o LassoCV. 

##### 1.1.3.1.2 Modelo de seleção baseado no critério de informação

    # Alternativamente, o estimador LassoLarsIC propõe o uso do critério de informação de Akaike (AIC) e do critério de informação de Bayes (BIC). É uma alternativa computacionalmente mais barata encontrar o valor ideal de alfa, pois o caminho de regularização é calculado apenas uma vez em vez de k + 1 vezes ao usar validação cruzada k-fold. No entanto, tais critérios precisam de uma estimativa adequada dos graus de liberdade da solução, são derivados para grandes amostras (resultados assintóticos) e assumem que o modelo está correto, ou seja, que os dados são gerados por este modelo. Eles também tendem a quebrar quando o problema está mal condicionado (mais variáveis do que amostras) 


    # Exemplos: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py

##### 1.1.3.1.3 Comparação com o parametro de regularização do SVM

    # A equivalência entre alfa e o parâmetro de regularização de SVM, C é dada por alfa = 1 / C ou alfa = 1 / (n_samples * C), dependendo do estimador e da função objetivo exata otimizada pelo modelo.