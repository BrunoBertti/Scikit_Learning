########## 1.2 Regressão e Classificação Ridge ##########

##### 1.1.2.1 Regressão

    # A regressão Ridge aborda alguns dos problemas do Mínimos Quadrados Ordinários, impondo uma panalidade no tamanho dos coeficientes. Os coefientes de Ridge minimizam a penalização da soma residual dos quadrados.
        
        # min || Xw - y||22 + alpha||w||22

    # A complexidade do parâmetro alpha >= 0 controla a quantidade de retração: quanto maior o valor de alpha, maior a quantidade de retração e, portanto, os coeficientes se tornam mais robustos à colinearidade.


    # Tal como acontece com outros modelos lineares, Ridge pegará em suas matrizes de método de ajuste X, y e armazenará os coeficientes w do modelo linear em seu membro coef_:
from sklearn import linear_model
reg = linear_model.Ridge(alpha=0.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.coef_)

print(reg.intercept_)


##### 1.1.2.2 Classificação

    # O Regressor Ridge tem uma variante classificadora: RidgeClassifier. Esse classificador primeiro converte os alvos binario em {-1, 1} e então trata o problema como uma tareda de regressão, otimizando o mesmo objetivo acima. A classe prevista corresponde ao sinal da previsão do regressor. Para a classificação multiclasse, o problema é tratado como uma regressão de múltiplas saídas, e a classe prevista corresponde à saída com o valor mais alto.

    # Pode parecer questionável usar uma perda de mínimos quadrados (panalizada) para treinar um modelo de classificação em vez das perdas logísticas ou dobradiças mais tradicionais. Entretanto, na prática, todos esses modelos podem levar a pontuações de validação cruzada semelhantes em termos de exatidão ou precisão/recall, enquanto a perda de mínimos quadrados é penalizada por usar RidgeClassifir permite uma escolha muito diferente dos solucionadores numéricos com perfis de desempenho computacional distintos.

    # O RidgeClassifier pode ser significativamente mais rápido do que o exemplo de LogisticRegression com um alto número de classes porque pode calcular a matrix de projeção (XtX)-X-1Xt somete uma vez

    # Esse claficador é algumas vezes referido como Mínimos Quadrados Support Vector Machines com um núcleo linear. (https://en.wikipedia.org/wiki/Least-squares_support-vector_machine)


    ## Exemplos:
    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html#sphx-glr-auto-examples-linear-model-plot-ridge-path-py
    ## https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
    ## https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py

##### 1.1.2.3 Complexidade Ridge

    # Esse método tem a mesma ordem de complexidade como Mínimos Quadrados Ordinários (https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

##### 1.1.2.4 Definindo o parâmetro de regularização: Validação Cruzada de deixar um de fora
 
    # Ridge implementa uma regressão rigde cpm validação cruzada embutida no parametro alpha. O objeto funciona da mesma maneira que GridSearchCv, exceto que o padrão é Validação Cruzada de deixar um de fora.
import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
print(reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1]))

print(reg.alpha_)

    # Especificar o valor do atributo cv acionará o uso de validação cruzada com GridSeachCV, por exemplo cv = 10 para validação cruzada de 10 vezes, em vez de Validação Cruzada Deixada de Fora.



    # Referências : “Notes on Regularized Least Squares”, Rifkin & Lippert (technical report, course slides) (http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf, https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf).