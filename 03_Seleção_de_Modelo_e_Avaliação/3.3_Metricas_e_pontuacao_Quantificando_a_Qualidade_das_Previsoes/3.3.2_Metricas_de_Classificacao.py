########## 3.3.2. Métricas de classificação ##########


    # O módulo sklearn.metrics implementa várias funções de perda, pontuação e utilidade para medir o desempenho da classificação. Algumas métricas podem exigir estimativas de probabilidade da classe positiva, valores de confiança ou valores de decisões binárias. A maioria das implementações permite que cada amostra forneça uma contribuição ponderada para a pontuação geral, por meio do parâmetro sample_weight.

    # Algumas delas são restritas ao caso de classificação binária: 


        # precision_recall_curve(y_true, probas_pred, *) Calcular pares de precisão-recall para diferentes limites de probabilidade.

        # roc_curve(y_true, y_score, *[, pos_label, ...]) Computar a característica operacional do receptor (ROC).

        # det_curve(y_true, y_score[, pos_label, ...]) Calcular taxas de erro para diferentes limites de probabilidade. 


    # Outros também funcionam no caso multiclasse: 


        # balanced_accuracy_score(y_true, y_pred, *[, ...]) Calcule a precisão balanceada.

        # cohen_kappa_score(y1, y2, *[, labels, ...]) Kappa de Cohen: uma estatística que mede a concordância entre os anotadores.

        # confusion_matrix(y_true, y_pred, *[, ...]) Calcular a matriz de confusão para avaliar a precisão de uma classificação.

        # hinge_loss(y_true, pred_decision, *[, ...]) Perda média da dobradiça (não regularizada).

        # matthews_corrcoef(y_true, y_pred, *[, ...]) Calcule o coeficiente de correlação de Matthews (MCC).

        # roc_auc_score(y_true, y_score, *[, average, ...]) Calcular a área sob a curva característica de operação do receptor (ROC AUC) a partir de pontuações de previsão.

        # top_k_accuracy_score(y_true, y_score, *[, ...]) Pontuação de classificação de precisão Top-k. 




    # Alguns também funcionam no caso multilabel: 

        # accuracy_score(y_true, y_pred, *[, ...]) Pontuação de classificação de precisão.

        # classification_report(y_true, y_pred, *[, ...]) Crie um relatório de texto mostrando as principais métricas de classificação.

        # f1_score(y_true, y_pred, *[, labels, ...]) Calcule a pontuação F1, também conhecida como F-score balanceado ou F-measure.

        # fbeta_score(y_true, y_pred, *, beta[, ...]) Calcule a pontuação F-beta.

        # hamming_loss(y_true, y_pred, *[, sample_weight]) Calcule a perda média de Hamming.

        # jaccard_score(y_true, y_pred, *[, labels, ...]) Pontuação do coeficiente de similaridade de Jaccard.

        # log_loss(y_true, y_pred, *[, eps, ...]) Perda de log, também conhecida como perda logística ou perda de entropia cruzada.

        # multilabel_confusion_matrix(y_true, y_pred, *) Calcule uma matriz de confusão para cada classe ou amostra.

        # precision_recall_fscore_support(y_true, ...) Precisão de cálculo, recall, F-measure e suporte para cada classe.

        # precision_score(y_true, y_pred, *[, labels, ...]) Calcule a precisão.

        # recall_score(y_true, y_pred, *[, labels, ...]) Calcular o recall.

        # roc_auc_score(y_true, y_score, *[, average, ...]) Calcular a área sob a curva característica de operação do receptor (ROC AUC) a partir de pontuações de previsão.

        # zero_one_loss(y_true, y_pred, *[, ...]) Perda de classificação zero-um. 



    # E alguns trabalham com problemas binários e multilabel (mas não multiclasse): 


        # average_precision_score(y_true, y_score, *) Calcular a precisão média (AP) das pontuações de previsão. 

    # Nas subseções a seguir, descreveremos cada uma dessas funções, precedidas por algumas notas sobre API comum e definição de métrica. 




##### 3.3.2.1. De binário para multiclasse e multilabel


    # Algumas métricas são essencialmente definidas para tarefas de classificação binária (por exemplo, f1_score, roc_auc_score). Nesses casos, por padrão, apenas o rótulo positivo é avaliado, assumindo por padrão que a classe positiva seja rotulada 1 (embora isso possa ser configurável através do parâmetro pos_label).

    # Ao estender uma métrica binária para problemas multiclasse ou multirótulo, os dados são tratados como uma coleção de problemas binários, um para cada classe. Há, então, várias maneiras de calcular a média de cálculos de métricas binárias no conjunto de classes, cada uma das quais pode ser útil em algum cenário. Onde disponível, você deve selecionar entre eles usando o parâmetro médio.

        # "macro" simplesmente calcula a média das métricas binárias, dando peso igual a cada classe. Em problemas em que classes infrequentes são importantes, a macro-média pode ser um meio de destacar seu desempenho. Por outro lado, a suposição de que todas as classes são igualmente importantes é muitas vezes falsa, de modo que a média macro enfatizará o desempenho tipicamente baixo em uma classe pouco frequente.

        # "ponderado" leva em conta o desequilíbrio de classe calculando a média de métricas binárias em que a pontuação de cada classe é ponderada por sua presença na amostra de dados verdadeiros.

        # "micro" dá a cada par de classe de amostra uma contribuição igual para a métrica geral (exceto como resultado do peso da amostra). Em vez de somar a métrica por classe, isso soma os dividendos e divisores que compõem as métricas por classe para calcular um quociente geral. A micromédia pode ser preferida em configurações multirrótulo, incluindo classificação multiclasse onde uma classe majoritária deve ser ignorada.

        # "amostras" aplica-se apenas a problemas com vários rótulos. Ele não calcula uma medida por classe, em vez disso calcula a métrica sobre as classes verdadeira e prevista para cada amostra nos dados de avaliação e retorna sua média (peso-amostra ponderada).

        # A seleção de average=None retornará um array com a pontuação de cada classe.

    # Enquanto os dados multiclasse são fornecidos para a métrica, como alvos binários, como uma matriz de rótulos de classe, os dados multilabel são especificados como uma matriz indicadora, na qual a célula [i, j] tem valor 1 se a amostra i tiver rótulo j e valor 0 caso contrário . 






##### 3.3.2.2. Pontuação de precisão


    # A função precision_score calcula a precisão, seja a fração (padrão) ou a contagem (normalize=False) das previsões corretas.

    # Na classificação multirótulo, a função retorna a precisão do subconjunto. Se todo o conjunto de rótulos previstos para uma amostra corresponder estritamente ao conjunto verdadeiro de rótulos, a precisão do subconjunto será 1,0; caso contrário, é 0,0.

    # Se \hat{y}_i for o valor previsto da i-ésima amostra e y_i for o valor verdadeiro correspondente, então a fração de previsões corretas sobre n_\text{amostras} é definida como 

        # \texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)

    # onde 1(x) é a função indicadora. 

import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
accuracy_score(y_true, y_pred, normalize=False)

    # No caso multilabel com indicadores de rótulo binário: 

accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

    # Exemplos:

    ##  See Test with permutations the significance of a classification score for an example of accuracy score usage using permutations of the dataset. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html#sphx-glr-auto-examples-model-selection-plot-permutation-tests-for-classification-py)


##### 3.3.2.3. Pontuação de precisão top-k

    # A função top_k_accuracy_score é uma generalização de precision_score. A diferença é que uma previsão é considerada correta desde que o rótulo verdadeiro esteja associado a uma das k pontuações mais altas previstas. exatidão_score é o caso especial de k = 1.

    # A função cobre os casos de classificação binária e multiclasse, mas não o caso multirrótulo.

    # Se \hat{f}_{i,j} for a classe prevista para a i-ésima amostra correspondente à j-ésima maior pontuação prevista e y_i for o valor verdadeiro correspondente, então a fração de previsões corretas sobre n_\text{ samples} é definido como 


        # \texttt{top-k accuracy}(y, \hat{f}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} \sum_{j=1}^{k} 1(\hat{f}_{i,j} = y_i)

    # onde k é o número de suposições permitidas e 1(x) é a função indicadora. 

import numpy as np
from sklearn.metrics import top_k_accuracy_score
y_true = np.array([0, 1, 2, 2])
y_score = np.array([[0.5, 0.2, 0.2],
                    [0.3, 0.4, 0.2],
                    [0.2, 0.4, 0.3],
                    [0.7, 0.2, 0.1]])
top_k_accuracy_score(y_true, y_score, k=2)

# A não normalização fornece o número de amostras classificadas "corretamente"
top_k_accuracy_score(y_true, y_score, k=2, normalize=False)





##### 3.3.2.4. Pontuação de precisão equilibrada


    # A função balanced_accuracy_score calcula a precisão balanceada, o que evita estimativas de desempenho infladas em conjuntos de dados desequilibrados. É a macromédia dos escores de recordação por classe ou, equivalentemente, acurácia bruta onde cada amostra é ponderada de acordo com a prevalência inversa de sua verdadeira classe. Assim, para conjuntos de dados balanceados, a pontuação é igual à precisão.

    # No caso binário, a precisão balanceada é igual à média aritmética de sensibilidade (taxa de verdadeiro positivo) e especificidade (taxa de verdadeiro negativo), ou a área sob a curva ROC com previsões binárias em vez de pontuações: 

        # \texttt{balanced-accuracy} = \frac{1}{2}\left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right )

    # Se o classificador tiver um desempenho igualmente bom em qualquer uma das classes, esse termo será reduzido à precisão convencional (ou seja, o número de previsões corretas dividido pelo número total de previsões).

    # Em contraste, se a precisão convencional estiver acima do acaso apenas porque o classificador tira vantagem de um conjunto de teste desbalanceado, então a precisão balanceada, conforme apropriado, cairá para \frac{1}{n\_classes}.

    # A pontuação varia de 0 a 1, ou quando ajustado=True é usado, é reescalonado para o intervalo \frac{1}{1 - n\_classes} para 1, inclusive, com desempenho em pontuação aleatória 0.

    # Se y_i for o valor verdadeiro da i-ésima amostra e w_i for o peso da amostra correspondente, ajustaremos o peso da amostra para:

        # \hat{w}_i = \frac{w_i}{\sum_j{1(y_j = y_i) w_j}}

    # onde 1(x) é a função indicadora. Dado o \hat{y}_i previsto para a amostra i, a precisão balanceada é definida como:

        # \texttt{precisão balanceada}(y, \hat{y}, w) = \frac{1}{\sum{\hat{w}_i}} \sum_i 1(\hat{y}_i = y_i) \ chapéu {w}_i


    # Com ajustado=Verdadeiro, a precisão balanceada relata o aumento relativo de \texttt{balanced-accuracy}(y, \mathbf{0}, w) =\frac{1}{n\_classes}. No caso binário, isso também é conhecido como *estatística J de Youden*, ou informação. 

    # Nota: A definição multiclasse aqui parece a extensão mais razoável da métrica usada na classificação binária, embora não haja certo consenso na literatura:
        # Nossa definição: [Mosley2013], [Kelleher2015] e [Guyon2015], onde [Guyon2015] adota a versão ajustada para garantir que as previsões aleatórias tenham uma pontuação de 0 e as previsões perfeitas tenham uma pontuação de 1..

        # Precisão balanceada de classe conforme descrito em [Mosley2013]: o mínimo entre a precisão e o recall para cada classe é calculado. Esses valores são então calculados sobre o número total de classes para obter a precisão balanceada.

        # Precisão equilibrada conforme descrito em [Urbanowicz2015]: a média de sensibilidade e especificidade é calculada para cada classe e, em seguida, calculada a média sobre o número total de classes. 

    
    ## Referências:

    ## Guyon2015(1,2) I. Guyon, K. Bennett, G. Cawley, H.J. Escalante, S. Escalera, T.K. Ho, N. Macià, B. Ray, M. Saeed, A.R. Statnikov, E. Viegas, Design of the 2015 ChaLearn AutoML Challenge, IJCNN 2015. (https://ieeexplore.ieee.org/document/7280767)

    ## Mosley2013(1,2) L. Mosley, A balanced approach to the multi-class imbalance problem, IJCV 2010. (https://lib.dr.iastate.edu/etd/13537/)

    ## Kelleher2015 John. D. Kelleher, Brian Mac Namee, Aoife D’Arcy, Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies, 2015. (https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics)

    ## Urbanowicz2015 Urbanowicz R.J., Moore, J.H. ExSTraCS 2.0: description and evaluation of a scalable learning classifier system, Evol. Intel. (2015) 8: 89. (https://doi.org/10.1007/s12065-015-0128-8)







##### 3.3.2.5. capa de Cohen

    # A função cohen_kappa_score calcula a estatística kappa de Cohen. Esta medida destina-se a comparar rotulagem por diferentes anotadores humanos, não um classificador versus uma verdade básica.

    # A pontuação kappa (consulte docstring) é um número entre -1 e 1. Pontuações acima de 0,8 são geralmente consideradas de boa concordância; zero ou menor significa nenhuma concordância (rótulos praticamente aleatórios).

    # As pontuações Kappa podem ser calculadas para problemas binários ou multiclasse, mas não para problemas com vários rótulos (exceto pelo cálculo manual de uma pontuação por rótulo) e não para mais de dois anotadores. 

from sklearn.metrics import cohen_kappa_score
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cohen_kappa_score(y_true, y_pred)


##### 3.3.2.6. Matriz de confusão




##### 3.3.2.7. Relatório de classificação




##### 3.3.2.8. Perda de Hamming





##### 3.3.2.9. Precisão, recall e medidas F





##### 3.3.2.9.1. Classificação binária

##### 3.3.2.9.2. Classificação multiclasse e multirótulo

##### 3.3.2.10. Pontuação do coeficiente de similaridade de Jaccard

##### 3.3.2.11. Perda da dobradiça

##### 3.3.2.12. Perda de registro

##### 3.3.2.13. Coeficiente de correlação de Matthews

##### 3.3.2.14. Matriz de confusão de vários rótulos

##### 3.3.2.15. Característica de operação do receptor (ROC)

##### 3.3.2.15.1. Caso binário

##### 3.3.2.15.2. Caso multiclasse

##### 3.3.2.15.3. Estojo multi-rótulo

##### 3.3.2.16. Compensação de erro de detecção (DET)

##### 3.3.2.17. Zero uma perda

##### 3.3.2.18. Perda de pontuação Brier 