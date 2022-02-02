########## 1.1.11 Regressão Logística ##########

    # A regressão logística, apesar do nome, é um modelo linear para classificação em vez de regressão. A regressão logística também é conhecida na literatura como regressão logit, classificação de entropia máxima (MaxEnt) ou classificador log-linear. Neste modelo, as probabilidades que descrevem os resultados possíveis de um único ensaio são modeladas usando uma função logística. 

    # A regressão logística é implementada em LogisticRegression. Esta implementação pode ajustar regressão logística binária, One-vs-Rest ou multinomial com regularização opcional l1, l2 ou Elastic-Net. 

    ## OBS: A regularização é aplicada por padrão, o que é comum no aprendizado de máquina, mas não nas estatísticas. Outra vantagem da regularização é que ela melhora a estabilidade numérica. Nenhuma regularização equivale a definir C com um valor muito alto. 

    # Como um problema de otimização, a regressão logística penalizada de classe binária l2 minimiza a seguinte função de custo:

        #min 1/2w^T * w + C somatório log(exp(-yi(X^Ti + c)) + 1)

    # Da mesma forma, a regressão logística regularizada l1 resolve o seguinte problema de otimização:

        #min ||w||i + C somatório log(exp(-yi(X^Ti + c)) + 1)

    # A regularização Elastic-Net é uma combinação de l1 e l2 e minimiza a seguinte função de custo:


        #min 1-p/2 w^Tw + p||w||1 + C somatório log(exp(-yi(X^Ti + c)) + 1)

    # onde p controla a força da regularização l1 vs. l2 regularização (corresponde ao parâmetro l1_ratio)

    # Observe que, nesta notação, é assumido que o alvo yi assume valores não conjunto -1,1 na tentativa i. Também podemos ver que Elastic-Net é equivalente a l1 quando p = 1 e equivalente a l2 quando p = 0 

    # Os solucionadores implementados na classe LogisticRegression são “liblinear”, “newton-cg”, “lbfgs”, “sag” e “saga”:

    # O solucionador “liblinear” usa um algoritmo de descida por coordenadas (CD) e conta com a excelente biblioteca C ++ LIBLINEAR, que é enviada com o scikit-learn. No entanto, o algoritmo CD implementado em liblinear não pode aprender um verdadeiro modelo multinomial (multiclasse); em vez disso, o problema de otimização é decomposto de forma “um contra o resto”, de forma que classificadores binários separados sejam treinados para todas as classes. Isso acontece nos bastidores, portanto, as instâncias de LogisticRegression que usam esse solucionador se comportam como classificadores multiclasse. Para a regularização l1, sklearn.svm.l1_min_c permite calcular o limite inferior de C para obter um modelo não “nulo” (todos os pesos dos recursos para zero).

    # Os solucionadores “lbfgs”, “sag” e “newton-cg” suportam apenas a regularização l2 ou nenhuma regularização e são encontrados para convergir mais rápido para alguns dados de alta dimensão. Definir multi_class como “multinomial” com esses solucionadores aprende um verdadeiro modelo de regressão logística multinomial 5, o que significa que suas estimativas de probabilidade devem ser mais bem calibradas do que a configuração padrão “um vs descanso”.

    # O solucionador “sag” usa Stochastic Average Gradient descendente 6. É mais rápido do que outros solucionadores para grandes conjuntos de dados, quando o número de amostras e o número de recursos são grandes.

    # O solucionador “saga” 7 é uma variante do “sag” que também suporta a penalidade não suave = "l1". Este é, portanto, o solucionador de escolha para regressão logística multinomial esparsa. É também o único solucionador que oferece suporte a penalty = "elasticnet".

    # O “lbfgs” é um algoritmo de otimização que se aproxima do algoritmo 8 de Broyden – Fletcher – Goldfarb – Shanno, que pertence a métodos quase Newton. O solucionador “lbfgs” é recomendado para uso para pequenos conjuntos de dados, mas para conjuntos de dados maiores, seu desempenho é prejudicado. 9

    # A tabela a seguir resume as penalidades suportadas por cada solucionador: 

    #                        solucionadores
    # Penalidades                   ‘liblinear’     ‘lbfgs’     ‘newton-cg’     ‘sag’      ‘saga’
    # Multinomial + L2 penalty          não           sim           sim          sim        sim
    # OVR + L2 penalty                  sim           sim           sim          sim        sim
    # Multinomial + L1 penalty          não           não           não          não        sim
    # OVR + L1 penalty                  sim           não           não          não        sim
    # Elastic-Net                       não           não           nã           não        sim
    # No penalty (‘none’)               não           sim           sim          sim        sim
    # Comportamentos 
    # Penalize the intercept (bad)      sim           não           não          não        não
    # Faster for large datasets         não           não           nã           sim        sim
    # Robust to unscaled datasets       sim           sim           sm           não        não



    # O solucionador “lbfgs” é usado por padrão por sua robustez. Para grandes conjuntos de dados, o solucionador “saga” é geralmente mais rápido. Para grandes conjuntos de dados, você também pode considerar o uso de SGDClassifier com perda de 'registro', que pode ser ainda mais rápido, mas requer mais ajuste. 

    ## Exemplos:
    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html#sphx-glr-auto-examples-linear-model-plot-logistic-l1-l2-sparsity-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#sphx-glr-auto-examples-linear-model-plot-logistic-path-py
    
    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html#sphx-glr-auto-examples-linear-model-plot-logistic-multinomial-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-mnist-py



    ## Diferenças do liblinear: Pode haver uma diferença nas pontuações obtidas entre LogisticRegression com solver = liblinear ou LinearSVC e a biblioteca liblinear externa diretamente, quando fit_intercept = False e o ajuste coef_ (ou) os dados a serem previstos são zeros. Isso ocorre porque para a (s) amostra (s) com função de decisão zero, LogisticRegression e LinearSVC prevêem a classe negativa, enquanto liblinear prevê a classe positiva. Observe que um modelo com fit_intercept = False e tendo muitas amostras com decisão_função zero, provavelmente é um modelo insuficiente e ruim e você é aconselhado a definir fit_intercept = True e aumentar o intercept_scaling. 

    ## OBS: Seleção de variáveis com regressão logística esparsa: Uma regressão logística com penalidade de l1 produz modelos esparsos e pode, portanto, ser usada para realizar a seleção de variáveis, conforme detalhado na seleção de variáveis com base em L1. (https://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection)

    ## OBS: estimativa do valor P : É possível obter os p-valores e intervalos de confiança dos coeficientes nos casos de regressão sem penalização. O pacote statsmodels <https://pypi.org/project/statsmodels/> oferece suporte nativo para isso. No sklearn, também se pode usar bootstrapping. 


    # LogisticRegressionCV implementa Regressão Logística com suporte de validação cruzada embutido, para encontrar os parâmetros C e l1_ratio ideais de acordo com o atributo de pontuação. Os solucionadores “newton-cg”, “sag”, “saga” e “lbfgs” são considerados mais rápidos para dados densos de alta dimensão, devido à inicialização a quente (consulte o Glossário). 



    ## Referências:

    ## Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 4.3.4

    ## Mark Schmidt, Nicolas Le Roux, and Francis Bach: Minimizing Finite Sums with the Stochastic Average Gradient. (https://hal.inria.fr/hal-00860051/document)

    ## Aaron Defazio, Francis Bach, Simon Lacoste-Julien: SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. (https://arxiv.org/abs/1407.0202)

    ## https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

    ## http://www.fuzihao.org/blog/2016/01/16/Comparison-of-Gradient-Descent-Stochastic-Gradient-Descent-and-L-BFGS/