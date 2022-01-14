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

    # A função confusion_matrix avalia a precisão da classificação calculando a matriz de confusão com cada linha correspondente à classe verdadeira (a Wikipédia e outras referências podem usar convenções diferentes para eixos).

    # Por definição, a entrada i, j em uma matriz de confusão é o número de observações realmente no grupo i, mas previsto para estar no grupo j. Aqui está um exemplo: 

from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)


    # ConfusionMatrixDisplay pode ser usado para representar visualmente uma matriz de confusão, conforme mostrado no exemplo da matriz de confusão, que cria a seguinte figura: 


        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    # O parâmetro normalize permite relatar proporções em vez de contagens. A matriz de confusão pode ser normalizada de 3 maneiras diferentes: 'pred', 'true' e 'all', que dividirá as contagens pela soma de cada coluna, linha ou toda a matriz, respectivamente. 

y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
confusion_matrix(y_true, y_pred, normalize='all')

    # Para problemas binários, podemos obter contagens de verdadeiros negativos, falsos positivos, falsos negativos e verdadeiros positivos da seguinte forma: 

y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
tn, fp, fn, tp

    

    ## Exemplos:
    ## See Confusion matrix for an example of using a confusion matrix to evaluate classifier output quality. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)

    ## See Recognizing hand-written digits for an example of using a confusion matrix to classify hand-written digits. (https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py)

    ## See Classification of text documents using sparse features for an example of using a confusion matrix to classify text documents. (https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py) 





##### 3.3.2.7. Relatório de classificação

    # A função class_report cria um relatório de texto mostrando as principais métricas de classificação. Aqui está um pequeno exemplo com target_names personalizados e rótulos inferidos: 

from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))


    ## Exemplos:

    ## See Recognizing hand-written digits for an example of classification report usage for hand-written digits. (https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py)

    ## See Classification of text documents using sparse features for an example of classification report usage for text documents. (https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py)

    ## See Parameter estimation using grid search with cross-validation for an example of classification report usage for grid search with nested cross-validation. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py)    


##### 3.3.2.8. Perda de Hamming


    # O hamming_loss calcula a perda de Hamming média ou a distância de Hamming entre dois conjuntos de amostras.

    # Se \hat{y}_j for o valor previsto para o j-ésimo rótulo de uma determinada amostra, y_j for o valor verdadeiro correspondente e n_\text{labels} for o número de classes ou rótulos, então a perda de Hamming L_{ Hamming} entre duas amostras é definido como:

        # L_{Hamming}(y, \hat{y}) = \frac{1}{n_\text{labels}} \sum_{j=0}^{n_\text{labels} - 1} 1(\hat{y}_j \not= y_j)

    # onde 1(x) é a função indicadora. 

from sklearn.metrics import hamming_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
hamming_loss(y_true, y_pred)

    # No caso multilabel com indicadores de rótulo binário: 

hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))

    # Nota: Na classificação multiclasse, a perda de Hamming corresponde à distância de Hamming entre y_true e y_pred que é semelhante à função Zero uma perda. No entanto, enquanto a perda zero-um penaliza conjuntos de previsão que não correspondem estritamente a conjuntos verdadeiros, a perda de Hamming penaliza rótulos individuais. Assim, a perda de Hamming, limitada superiormente pela perda zero-um, está sempre entre zero e um, inclusive; e prever um subconjunto ou superconjunto adequado dos rótulos verdadeiros dará uma perda de Hamming entre zero e um, exclusivo 



##### 3.3.2.9. Precisão, recall e medidas F

    # Intuitivamente, precisão é a habilidade do classificador de não rotular como positiva uma amostra que é negativa, e recall é a habilidade do classificador de encontrar todas as amostras positivas.

    # A medida F (medidas F_\beta e F_1) pode ser interpretada como uma média harmônica ponderada da precisão e do recall. Uma medida F_\beta atinge seu melhor valor em 1 e sua pior pontuação em 0. Com \beta = 1, F_\beta e F_1 são equivalentes, e o recall e a precisão são igualmente importantes.

    # A precisão_recall_curve calcula uma curva de precisão-recall a partir do rótulo de verdade e uma pontuação dada pelo classificador variando um limite de decisão.

    # A função average_precision_score calcula a precisão média (AP) das pontuações de previsão. O valor está entre 0 e 1 e maior é melhor. AP é definido como

        # \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    # onde P_n e R_n são a precisão e a rechamada no enésimo limiar. Com previsões aleatórias, o AP é a fração de amostras positivas.

    # As referências [Manning2008] e [Everingham2010] apresentam variantes alternativas de AP que interpolam a curva de precisão-recall. Atualmente, average_precision_score não implementa nenhuma variante interpolada. As referências [Davis2006] e [Flach2015] descrevem por que uma interpolação linear de pontos na curva de recuperação de precisão fornece uma medida excessivamente otimista do desempenho do classificador. Esta interpolação linear é usada ao calcular a área sob a curva com a regra trapezoidal em auc.

    # Várias funções permitem analisar a precisão, o recall e a pontuação das medidas F: 


        # average_precision_score(y_true, y_score, *) Calcular a precisão média (AP) das pontuações de previsão.

        # f1_score(y_true, y_pred, *[, labels, ...]) Calcule a pontuação F1, também conhecida como F-score balanceado ou F-measure.

        # fbeta_score(y_true, y_pred, *, beta[, ...]) Calcule a pontuação F-beta.

        # precision_recall_curve(y_true, probas_pred, *) Calcular pares de precisão-recall para diferentes limites de probabilidade.

        # precision_recall_fscore_support(y_true, ...) Precisão de cálculo, recall, F-measure e suporte para cada

        # precision_score(y_true, y_pred, *[, labels, ...]) Calcule a precisão.

        # recall_score(y_true, y_pred, *[, labels, ...]) Calcular o recall. 


    # Observe que a função precision_recall_curve é restrita ao caso binário. A função average_precision_score funciona apenas na classificação binária e no formato de indicador multilabel. As funções PredictionRecallDisplay.from_estimator e PredictionRecallDisplay.from_predictions plotarão a curva de recuperação de precisão da seguinte maneira. 

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-the-precision-recall-curve




    ## Exemplos:

    ## See Classification of text documents using sparse features for an example of f1_score usage to classify text documents. (https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py)

    ## See Parameter estimation using grid search with cross-validation for an example of precision_score and recall_score usage to estimate parameters using grid search with nested cross-validation. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py)

    ## See Precision-Recall for an example of precision_recall_curve usage to evaluate classifier output quality. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py)




    ## Referências:

    ## Manning2008 C.D. Manning, P. Raghavan, H. Schütze, Introduction to Information Retrieval, 2008. (https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)

    ## Everingham2010 M. Everingham, L. Van Gool, C.K.I. Williams, J. Winn, A. Zisserman, The Pascal Visual Object Classes (VOC) Challenge, IJCV 2010. (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf)

    ## Davis2006 J. Davis, M. Goadrich, The Relationship Between Precision-Recall and ROC Curves, ICML 2006. (https://www.biostat.wisc.edu/~page/rocpr.pdf)

    ## Flach2015 P.A. Flach, M. Kull, Precision-Recall-Gain Curves: PR Analysis Done Right, NIPS 2015. (https://papers.nips.cc/paper/5867-precision-recall-gain-curves-pr-analysis-done-right.pdf)



##### 3.3.2.9.1. Classificação binária

    # Em uma tarefa de classificação binária, os termos “positivo” e “negativo” referem-se à previsão do classificador, e os termos “verdadeiro” e “falso” referem-se a se essa previsão corresponde ao julgamento externo ( às vezes conhecido como a ''observação''). Dadas essas definições, podemos formular a seguinte tabela: 



#                               Aula real (observação) 
#
#
# Classe prevista           tp (verdadeiro positivo)        fp (falso positivo)
# (expectativa)             Resultado correto               Resultado inesperado 
#
#                           fn (falso negativo)             tn (verdadeiro negativo)
#                           Resultado ausente               Ausência correta de resultado 
#


    # Neste contexto, podemos definir as noções de precisão, recall e F-measure: 


        # \text{precision} = \frac{tp}{tp + fp},


        # \text{recall} = \frac{tp}{tp + fn},


        # F_\beta = (1 + \beta^2) \frac{\text{precision} \times \text{recall}}{\beta^2 \text{precision} + \text{recall}}.



    # Aqui estão alguns pequenos exemplos de classificação binária: 

from sklearn import metrics
y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]
metrics.precision_score(y_true, y_pred)
metrics.recall_score(y_true, y_pred)

metrics.f1_score(y_true, y_pred)

metrics.fbeta_score(y_true, y_pred, beta=0.5)

metrics.fbeta_score(y_true, y_pred, beta=1)

metrics.fbeta_score(y_true, y_pred, beta=2)

metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5)




import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, threshold = precision_recall_curve(y_true, y_scores)
precision
recall
threshold
average_precision_score(y_true, y_scores)




##### 3.3.2.9.2. Classificação multiclasse e multirótulo

    # Na tarefa de classificação multiclasse e multirótulo, as noções de precisão, rechamada e medidas F podem ser aplicadas a cada rótulo independentemente. Existem algumas maneiras de combinar resultados entre rótulos, especificados pelo argumento average para as funções average_precision_score (somente multilabel), f1_score, fbeta_score, precision_recall_fscore_support, precision_score e recall_score, conforme descrito acima. Observe que, se todos os rótulos forem incluídos, a média “micro” em uma configuração multiclasse produzirá precisão, rechamada e F que são todos idênticos à exatidão. Observe também que a média “ponderada” pode produzir um F-score que não está entre precisão e recuperação. 

    # Para tornar isso mais explícito, considere a seguinte notação: 

        # y o conjunto de pares previstos (amostra, rótulo)


        # \hat{y} o conjunto de pares verdadeiros (amostra, rótulo)

        # L o conjunto de rótulos

        # S o conjunto de amostras


        # y_s o subconjunto de y com amostra s, ou seja, y_s := \left\{(s', l) \in y | s' = s\direita\}


        # y_l o subconjunto de y com rótulo l

        # da mesma forma, \hat{y}_s e \hat{y}_l são subconjuntos de \hat{y}


        # P(A, B) := \frac{\esquerda| A \cap B \right|}{\left|A\right|} para alguns conjuntos A e B


        # R(A, B) := \frac{\esquerda| A \cap B \right|}{\left|B\right|} (As convenções variam no tratamento de B = \emptyset; esta implementação usa R(A, B):=0 e similar para P.)

        # F_\beta(A, B) := \left(1 + \beta^2\right) \frac{P(A, B) \times R(A, B)}{\beta^2 P(A, B) ) + R(A, B)} 


    # Então as métricas são definidas como: 


        ########## TABELA #########


from sklearn import metrics
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
metrics.precision_score(y_true, y_pred, average='macro')
metrics.recall_score(y_true, y_pred, average='micro')
metrics.f1_score(y_true, y_pred, average='weighted')
metrics.fbeta_score(y_true, y_pred, average='macro', beta=0.5)
metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5, average=None)


    # Para classificação multiclasse com “classe negativa”, é possível excluir alguns rótulos: 

metrics.recall_score(y_true, y_pred, labels=[1, 2], average='micro')
# excluindo 0, nenhum rótulo foi recuperado corretamente 

    # Da mesma forma, rótulos não presentes na amostra de dados podem ser considerados na média macro.

metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro')





##### 3.3.2.10. Pontuação do coeficiente de similaridade de Jaccard


    # A função jaccard_score calcula a média dos coeficientes de similaridade de Jaccard, também chamado de índice de Jaccard, entre pares de conjuntos de rótulos.

    # O coeficiente de similaridade de Jaccard das i-ésimas amostras, com um conjunto de rótulos de verdade y_i e conjunto de rótulos previsto \hat{y}_i, é definido como

    # J(y_i, \hat{y}_i) = \frac{|y_i \cap \hat{y}_i|}{|y_i \cup \hat{y}_i|}.

    # jaccard_score funciona como precision_recall_fscore_support como uma medida ingenuamente definida aplicada nativamente a alvos binários e estendida para aplicar a multilabel e multiclass através do uso de média (veja acima).

    # No caso binário: 

import numpy as np
from sklearn.metrics import jaccard_score
y_true = np.array([[0, 1, 1], [1, 1, 0]])
y_pred = np.array([[1, 1, 1], [1, 0, 0]])
jaccard_score(y_true[0], y_pred)


    # No caso multilabel com indicadores de rótulo binário: 

jaccard_score(y_true, y_pred, average='samples')
jaccard_score(y_true, y_pred, average='macro')
jaccard_score(y_true, y_pred, average=N)

    # Problemas multiclasse são binarizados e tratados como o problema multirótulo correspondente: 

y_pred = [0, 2, 1, 2]
y_true = [0, 1, 2, 2]
jaccard_score(y_true, y_pred, average=None)
jaccard_score(y_true, y_pred, average='macro')
jaccard_score(y_true, y_pred, average='mic')

##### 3.3.2.11. Perda da dobradiça

    # A função de perda de dobradiça calcula a distância média entre o modelo e os dados usando a perda de dobradiça, uma métrica unilateral que considera apenas os erros de previsão. (A perda de dobradiça é usada em classificadores de margem máxima, como máquinas de vetor de suporte.)

    # Se os rótulos forem codificados com +1 e -1, y: é o valor verdadeiro e w são as decisões previstas como saída pela função_decisão, então a perda de dobradiça é definida como: 

        # L_\text{Hinge}(y, w) = \max\left\{1 - wy, 0\right\} = \left|1 - wy\right|_+


    # Se houver mais de dois rótulos, a dobradiça_loss usa uma variante multiclasse devido a Crammer & Singer. Aqui está o papel que o descreve.

    # Se y_w é a decisão prevista para rótulo verdadeiro e y_t é o máximo das decisões previstas para todos os outros rótulos, onde as decisões previstas são emitidas pela função de decisão, então a perda de dobradiça multiclasse é definida por: 

        # L_\text{Hinge}(y_w, y_t) = \max\left\{1 + y_t - y_w, 0\right\}

    # Aqui um pequeno exemplo demonstrando o uso da função dobradiça_loss com um classificador svm em um problema de classe binária: 

from sklearn import svm
from sklearn.metrics import hinge_loss
X = [[0], [1]]
y = [-1, 1]
est = svm.LinearSVC(random_state=0)
est.fit(X, y)
LinearSVC(random_state=0)
pred_decision = est.decision_function([[-2], [3], [0.5]])
pred_decision
hinge_loss([-1, 1, 1], pred_decision)


    # Aqui está um exemplo demonstrando o uso da função dobradiça_loss com um classificador svm em um problema multiclasse: 

X = np.array([[0], [1], [2], [3]])
Y = np.array([0, 1, 2, 3])
labels = np.array([0, 1, 2, 3])
est = svm.LinearSVC()
est.fit(X, Y)
LinearSVC()
pred_decision = est.decision_function([[-1], [2], [3]])
y_true = [0, 2, 3]
hinge_loss(y_true, pred_decision, labels=labels)




##### 3.3.2.12. Perda de registro


    # A perda de log, também chamada de perda de regressão logística ou perda de entropia cruzada, é definida em estimativas de probabilidade. É comumente usado em regressão logística (multinomial) e redes neurais, bem como em algumas variantes de maximização de expectativa, e pode ser usado para avaliar as saídas de probabilidade (predict_proba) de um classificador em vez de suas previsões discretas.

    # Para classificação binária com um rótulo verdadeiro y \in \{0,1\} e uma estimativa de probabilidade p = \operatorname{Pr}(y = 1), a perda logarítmica por amostra é a probabilidade logarítmica negativa do classificador dado o rótulo verdadeiro: 

        # L_{\log}(y, p) = -\log \operatorname{Pr}(y|p) = -(y \log (p) + (1 - y) \log (1 - p))


    # Isso se estende ao caso multiclasse da seguinte maneira. Deixe que os rótulos verdadeiros para um conjunto de amostras sejam codificados como uma matriz indicadora binária Y de 1 de K, ou seja, y_{i,k} = 1 se a amostra i tiver o rótulo k retirado de um conjunto de K rótulos. Seja P uma matriz de estimativas de probabilidade, com p_{i,k} = \operatorname{Pr}(y_{i,k} = 1). Então a perda de log de todo o conjunto é 

        # L_{\log}(Y, P) = -\log \operatorname{Pr}(Y|P) = - \frac{1}{N} \sum_{i=0}^{N-1} \sum_{k=0}^{K-1} y_{i,k} \log p_{i,k}

    # Para ver como isso generaliza a perda de log binário fornecida acima, observe que, no caso binário, p_{i,0} = 1 - p_{i,1} e y_{i,0} = 1 - y_{i,1} , então expandir a soma interna sobre y_{i,k} \in \{0,1\} dá a perda de log binário.

    # A função log_loss calcula a perda de log dada uma lista de rótulos de verdade e uma matriz de probabilidade, conforme retornado pelo método predict_proba de um estimador. 

from sklearn.metrics import log_loss
y_true = [0, 0, 1, 1]
y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
log_loss(y_true, y_pred)

    # O primeiro [.9, .1] em y_pred denota 90% de probabilidade de que a primeira amostra tenha o rótulo 0. A perda de log não é negativa. 


##### 3.3.2.13. Coeficiente de correlação de Matthews

##### 3.3.2.14. Matriz de confusão de vários rótulos

##### 3.3.2.15. Característica de operação do receptor (ROC)

##### 3.3.2.15.1. Caso binário

##### 3.3.2.15.2. Caso multiclasse

##### 3.3.2.15.3. Estojo multi-rótulo

##### 3.3.2.16. Compensação de erro de detecção (DET)

##### 3.3.2.17. Zero uma perda

##### 3.3.2.18. Perda de pontuação Brier 