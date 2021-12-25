########## 1.16.3. Uso  ##########

    # A classe CalibratedClassifierCV é usada para calibrar um classificador.

    # CalibratedClassifierCV usa uma abordagem de validação cruzada para garantir que dados imparciais sejam sempre usados ​​para ajustar o calibrador. Os dados são divididos em k (train_set, test_set) pares (conforme determinado por cv). Quando ensemble = True (padrão), o seguinte procedimento é repetido independentemente para cada divisão de validação cruzada: um clone de base_estimator é treinado primeiro no subconjunto de treino. Em seguida, suas previsões no subconjunto de teste são usadas para ajustar um calibrador (um regressor sigmóide ou isotônico). Isso resulta em um conjunto de k (classificador, calibrador) pares onde cada calibrador mapeia a saída de seu classificador correspondente em [0, 1]. Cada par é exposto no atributo calibrated_classifiers_, onde cada entrada é um classificador calibrado com um método predict_proba que produz probabilidades calibradas. A saída de Predict_proba para a instância CalibratedClassifierCV principal corresponde à média das probabilidades previstas dos estimadores k na lista calibrated_classifiers_. A saída da previsão é a classe que tem a maior probabilidade.

    # Quando ensemble = False, a validação cruzada é usada para obter previsões "imparciais" para todos os dados, via cross_val_predict. Essas previsões imparciais são então usadas para treinar o calibrador. O atributo calibrated_classifiers_ consiste em apenas um par (classificador, calibrador) onde o classificador é o base_estimator treinado em todos os dados. Neste caso, a saída de Predict_proba para CalibratedClassifierCV são as probabilidades preditas obtidas de um único par (classificador, calibrador).

    # A principal vantagem de ensemble = True é se beneficiar do efeito de ensembling tradicional (semelhante ao metaestimador de Bagging). O conjunto resultante deve ser bem calibrado e ligeiramente mais preciso do que com conjunto = Falso. A principal vantagem de usar ensemble = False é computacional: reduz o tempo de ajuste geral treinando apenas um único par de classificador e calibrador de base, diminui o tamanho do modelo final e aumenta a velocidade de previsão.

    # Alternativamente, um classificador já instalado pode ser calibrado definindo cv = "prefit". Nesse caso, os dados não são divididos e todos são usados ​​para ajustar o regressor. Cabe ao usuário certificar-se de que os dados usados ​​para ajustar o classificador sejam separados dos dados usados ​​para ajustar o regressor.

    # sklearn.metrics.brier_score_loss pode ser usado para avaliar o quão bem um classificador está calibrado. No entanto, essa métrica deve ser usada com cuidado porque um escore de Brier mais baixo nem sempre significa um modelo mais bem calibrado. Isso ocorre porque a métrica de pontuação de Brier é uma combinação de perda de calibração e perda de refinamento. A perda de calibração é definida como o desvio médio quadrático das probabilidades empíricas derivadas da inclinação dos segmentos ROC. A perda de refinamento pode ser definida como a perda ótima esperada, medida pela área sob a curva de custo ótima. Como a perda de refinamento pode mudar independentemente da perda de calibração, uma pontuação de Brier mais baixa não significa necessariamente um modelo melhor calibrado.

    # CalibratedClassifierCV suporta o uso de dois regressores de 'calibração': 'sigmóide' e 'isotônico'. 

##### 1.16.3.1. Sigmóide

    # O regressor sigmóide é baseado no modelo logístico 3 de Platt: 

        # p(y_i = 1 | f_i) = \frac{1}{1 + \exp(A f_i + B)}

    # onde y_i é o verdadeiro rótulo da amostra ie f_i é a saída do classificador não calibrado para a amostra i. A e B são números reais a serem determinados ao ajustar o regressor via máxima verossimilhança.


    # O método sigmóide assume que a curva de calibração pode ser corrigida aplicando uma função sigmóide às previsões brutas. Esta suposição foi empiricamente justificada no caso de Support Vector Machines com funções de kernel comuns em vários conjuntos de dados de benchmark na seção 2.1 de Platt 1999 3, mas não necessariamente se aplica em geral. Além disso, o modelo logístico funciona melhor se o erro de calibração for simétrico, o que significa que a saída do classificador para cada classe binária é normalmente distribuída com a mesma variância 6. Isso pode ser um problema para problemas de classificação altamente desequilibrados, onde as saídas não têm igual variância.

    # Em geral, esse método é mais eficaz quando o modelo não calibrado é pouco confiável e tem erros de calibração semelhantes para saídas altas e baixas. 

##### 1.16.3.2. Isotônico

    # O método 'isotônico' se ajusta a um regressor isotônico não paramétrico, que produz uma função não decrescente em etapas (consulte sklearn.isotonic). Minimiza: 

        # \sum_{i=1}^{n} (y_i - \hat{f}_i)^2

    # sujeito a \ hat {f} _i> = \ hat {f} _j sempre que f_i> = f_j. y_i é o verdadeiro rótulo da amostra i e \ hat {f} _i é a saída do classificador calibrado para a amostra i (ou seja, a probabilidade calibrada). Este método é mais geral quando comparado a "sigmóide", pois a única restrição é que a função de mapeamento está aumentando monotonicamente. É, portanto, mais poderoso, pois pode corrigir qualquer distorção monotônica do modelo não calibrado. No entanto, é mais sujeito a overfitting, especialmente em pequenos conjuntos de dados 5.


    # No geral, 'isotônico' terá um desempenho tão bom ou melhor do que 'sigmóide' quando houver dados suficientes (mais de ~ 1000 amostras) para evitar sobreajuste 1. 

##### 1.16.3.3. Suporte multiclasse

    # Ambos os regressores isotônicos e sigmóides suportam apenas dados unidimensionais (por exemplo, saída de classificação binária), mas são estendidos para classificação multiclasse se o base_estimator suportar previsões multiclasse. Para previsões multiclasse, CalibratedClassifierCV calibra para cada classe separadamente em um estilo OneVsRestClassifier 4. Ao prever probabilidades, as probabilidades calibradas para cada classe são previstas separadamente. Como essas probabilidades não somam necessariamente um, é realizado um pós-processamento para normalizá-las. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py

    ## https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html#sphx-glr-auto-examples-calibration-plot-calibration-multiclass-py

    ## https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py

    ## https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#sphx-glr-auto-examples-calibration-plot-compare-calibration-py




    ## Referências:

    ## 1(1,2,3) Predicting Good Probabilities with Supervised Learning, A. Niculescu-Mizil & R. Caruana, ICML 2005 (https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)

    ## 2 On the combination of forecast probabilities for consecutive precipitation periods. Wea. Forecasting, 5, 640–650., Wilks, D. S., 1990a (https://journals.ametsoc.org/waf/article/5/4/640/40179)

    ## 3(1,2) Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. J. Platt, (1999) (https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf)

    ## Transforming Classifier Scores into Accurate Multiclass Probability Estimates. B. Zadrozny & C. Elkan, (KDD 2002) (https://dl.acm.org/doi/pdf/10.1145/775047.775151)

    ## Predicting accurate probabilities with a ranking loss. Menon AK, Jiang XJ, Vembu S, Elkan C, Ohno-Machado L. Proc Int Conf Mach Learn. 2012;2012:703-710 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180410/)

    ## Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration Kull, M., Silva Filho, T. M., & Flach, P. (2017). (https://projecteuclid.org/euclid.ejs/1513306867)