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

    # Observe que, nesta notação, é assumido que o alvo yi assume valores no conjunto -1,1 na tentativa i. Também podemos ver que Elastic-Net é equivalente a l1 quando p = 1 e equivalente a l2 quando p = 0 

    # Os solucionadores implementados na classe LogisticRegression são “liblinear”, “newton-cg”, “lbfgs”, “sag” e “saga”:

    # O solucionador “liblinear” usa um algoritmo de descida por coordenadas (CD) e conta com a excelente biblioteca C ++ LIBLINEAR, que é enviada com o scikit-learn. No entanto, o algoritmo CD implementado em liblinear não pode aprender um verdadeiro modelo multinomial (multiclasse); em vez disso, o problema de otimização é decomposto de forma “um contra o resto”, de forma que classificadores binários separados sejam treinados para todas as classes. Isso acontece nos bastidores, portanto, as instâncias de LogisticRegression que usam esse solucionador se comportam como classificadores multiclasse. Para a regularização l1, sklearn.svm.l1_min_c permite calcular o limite inferior de C para obter um modelo não “nulo” (todos os pesos dos recursos para zero).

    # Os solucionadores “lbfgs”, “sag” e “newton-cg” suportam apenas a regularização l2 ou nenhuma regularização e são encontrados para convergir mais rápido para alguns dados de alta dimensão. Definir multi_class como “multinomial” com esses solucionadores aprende um verdadeiro modelo de regressão logística multinomial 5, o que significa que suas estimativas de probabilidade devem ser mais bem calibradas do que a configuração padrão “um vs descanso”.

    # O solucionador “sag” usa Stochastic Average Gradient descendente 6. É mais rápido do que outros solucionadores para grandes conjuntos de dados, quando o número de amostras e o número de recursos são grandes.

    # O solucionador “saga” 7 é uma variante do “sag” que também suporta a penalidade não suave = "l1". Este é, portanto, o solucionador de escolha para regressão logística multinomial esparsa. É também o único solucionador que oferece suporte a penalty = "elasticnet".

    # O “lbfgs” é um algoritmo de otimização que se aproxima do algoritmo 8 de Broyden – Fletcher – Goldfarb – Shanno, que pertence a métodos quase Newton. O solucionador “lbfgs” é recomendado para uso para pequenos conjuntos de dados, mas para conjuntos de dados maiores, seu desempenho é prejudicado. 9

    # A tabela a seguir resume as penalidades suportadas por cada solucionador: 

    #                        solucionadores
    # Penalidades                   ‘liblinear’     ‘lbfgs’     ‘newton-cg’     ‘sag’      ‘saga’
    # Multinomial + L2 penalty          no            yes           yes          yes        yes
    # OVR + L2 penalty                  yes           yes           yes          yes        yes
    # Multinomial + L1 penalty          no            no            no           no         yes
    # OVR + L1 penalty                  yes           no            no           no         yes
    # Elastic-Net                       no            no            no           no         yes
    # No penalty (‘none’)               no            yes           yes          yes        yes
    # Comportamentos 
    # Penalize the intercept (bad)      yes           no            no           no         no
    # Faster for large datasets         no            no            no           yes        yes
    # Robust to unscaled datasets       yes           yes           yes          no         no

