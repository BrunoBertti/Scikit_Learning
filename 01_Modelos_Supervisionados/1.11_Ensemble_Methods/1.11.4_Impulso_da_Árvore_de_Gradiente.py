########## 1.11.4 Reforço da árvore de gradiente ##########


    # Gradient Tree Boosting ou Gradient Boosted Decision Trees (GBDT) é uma generalização do boost para funções de perda diferenciáveis arbitrárias. GBDT é um procedimento de prateleira preciso e eficaz que pode ser usado para problemas de regressão e classificação em uma variedade de áreas, incluindo classificação de pesquisa na Web e ecologia.

    # O módulo sklearn.ensemble fornece métodos para classificação e regressão por meio de árvores de decisão incrementadas por gradiente. 


    # OBS: Scikit-learn 0.21 apresenta duas novas implementações de árvores de aumento de gradiente, ou seja, HistGradientBoostingClassifier e HistGradientBoostingRegressor, inspirado por LightGBM (Consulte [LightGBM]).
    # Esses estimadores baseados em histograma podem ser ordens de magnitude mais rápidos do que GradientBoostingClassifier e GradientBoostingRegressor quando o número de amostras é maior do que dezenas de milhares de amostras.
    # Eles também têm suporte integrado para valores ausentes, o que evita a necessidade de um imputador.
    # Esses estimadores são descritos em mais detalhes abaixo em Histogram-Based Gradient Boosting.
    # O guia a seguir concentra-se em GradientBoostingClassifier e GradientBoostingRegressor, que podem ser preferidos para tamanhos de amostra pequenos, pois o binning pode levar a pontos de divisão que são muito aproximados nesta configuração.

    # O uso e os parâmetros de GradientBoostingClassifier e GradientBoostingRegressor são descritos a seguir. Os 2 parâmetros mais importantes desses estimadores são n_estimators e learning_rate. 




##### 1.11.4.1. Classificação

    # GradientBoostingClassifier oferece suporte à classificação binária e multiclasse. O exemplo a seguir mostra como ajustar um classificador de aumento de gradiente com 100 tocos de decisão como alunos fracos: 

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)


    # O número de alunos fracos (ou seja, árvores de regressão) é controlado pelo parâmetro n_estimators; O tamanho de cada árvore pode ser controlado configurando a profundidade da árvore via max_depth ou configurando o número de nós folha via max_leaf_nodes. O learning_rate é um hiperparâmetro no intervalo (0,0, 1,0] que controla o sobreajuste via redução.

    # OBS: A classificação com mais de 2 classes requer a indução de árvores de regressão n_classes a cada iteração, portanto, o número total de árvores induzidas é igual a n_classes * n_estimators. Para conjuntos de dados com um grande número de classes, é altamente recomendável usar HistGradientBoostingClassifier como uma alternativa para GradientBoostingClassifier. 

##### 1.11.4.2. Regressão

    # GradientBoostingRegressor oferece suporte a várias funções de perda diferentes para regressão que podem ser especificadas por meio do argumento loss; a função de perda padrão para regressão é erro quadrado ('squared_error'). 


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='squared_error').fit(X_train, y_train)
mean_squared_error(y_test, est.predict(X_test))

    # A figura abaixo mostra os resultados da aplicação de GradientBoostingRegressor com perda de mínimos quadrados e 500 alunos básicos ao conjunto de dados de diabetes (sklearn.datasets.load_diabetes). O gráfico à esquerda mostra o trem e o erro de teste em cada iteração. O train_error em cada iteração é armazenado no atributo train_score_ do modelo de aumento de gradiente. O erro de teste em cada iteração pode ser obtido por meio do método staged_predict, que retorna um gerador que produz as previsões em cada estágio. Parcelas como essas podem ser usadas para determinar o número ideal de árvores (ou seja, n_estimators) por parada antecipada. 

        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

    
    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-oob-py

##### 1.11.4.3. Adequando alunos mais fracos

    # GradientBoostingRegressor e GradientBoostingClassifier suportam warm_start = True, o que permite adicionar mais estimadores a um modelo já ajustado. 

_ = est.set_params(n_estimators=200, warm_start=True)  # definir warm_start e novo nr de árvores
_ = est.fit(X_train, y_train) # caber 100 árvores adicionais para est 
mean_squared_error(y_test, est.predict(X_test))



##### 1.11.4.4. Controlando o tamanho da árvore

    # O tamanho dos alunos com base na árvore de regressão define o nível de interações de variáveis ​​que podem ser capturadas pelo modelo de aumento de gradiente. Em geral, uma árvore de profundidade h pode capturar interações da ordem h. Existem duas maneiras de controlar o tamanho das árvores de regressão individuais.

    # Se você especificar max_depth = h, então árvores binárias completas de profundidade h serão cultivadas. Essas árvores terão (no máximo) 2 ** h nós de folhas e 2 ** h - 1 nós de divisão.

    # Como alternativa, você pode controlar o tamanho da árvore especificando o número de nós folha por meio do parâmetro max_leaf_nodes. Nesse caso, as árvores serão cultivadas usando a pesquisa best-first, onde os nós com a maior melhoria na impureza serão expandidos primeiro. Uma árvore com max_leaf_nodes = k tem k - 1 nós de divisão e, portanto, pode modelar interações de até max_leaf_nodes - 1.

    # Descobrimos que max_leaf_nodes = k fornece resultados comparáveis ​​a max_depth = k-1, mas é significativamente mais rápido para treinar às custas de um erro de treinamento ligeiramente maior. O parâmetro max_leaf_nodes corresponde à variável J no capítulo sobre aumento de gradiente em [F2001] e está relacionado ao parâmetro interação.depth no pacote gbm de R, onde max_leaf_nodes == interação.depth + 1. 

##### 1.11.4.5. Formulação matemática

    # Primeiro apresentamos o GBRT para regressão e, em seguida, detalhamos o caso de classificação. 


##### 1.11.4.5.1. Regressão

    # Os regressores GBRT são modelos aditivos cuja previsão y_i para uma determinada entrada x_i tem a seguinte forma:

        # \ hat {y_i} = F_M (x_i) = \ sum_ {m = 1} ^ {M} h_m (x_i)

    # onde h_m são estimadores chamados alunos fracos no contexto de boost. O Gradient Tree Boosting usa regressores de árvore de decisão de tamanho fixo como alunos fracos. A constante M corresponde ao parâmetro n_estimators.

    # Semelhante a outros algoritmos de impulso, um GBRT é construído de forma gananciosa:

        # F_m (x) = F_ {m-1} (x) + h_m (x),


    # onde a árvore recém-adicionada h_m é ajustada a fim de minimizar a soma das perdas L_m, dado o conjunto anterior F_ {m-1}:


        # h_m = \ arg \ min_ {h} L_m = \ arg \ min_ {h} \ sum_ {i = 1} ^ {n}
        # l (y_i, F_ {m-1} (x_i) + h (x_i)),



    # onde l (y_i, F (x_i)) é definido pelo parâmetro de perda, detalhado na próxima seção.


    # Por padrão, o modelo inicial F_ {0} é escolhido como a constante que minimiza a perda: para uma perda de mínimos quadrados, esta é a média empírica dos valores alvo. O modelo inicial também pode ser especificado por meio do argumento init.


    # Usando uma aproximação de Taylor de primeira ordem, o valor de pode ser aproximado da seguinte forma:

        # l (y_i, F_ {m-1} (x_i) + h_m (x_i)) \ aprox
        # l (y_i, F_ {m-1} (x_i))
        # + h_m (x_i)
        # \ left [\ frac {\ partial l (y_i, F (x_i))} {\ partial F (x_i)} \ right] _ {F = F_ {m - 1}}.


    # OBS: Resumidamente, uma aproximação de Taylor de primeira ordem diz que l (z) \ approx l (a) + (z - a) \ frac {\ parcial l (a)} {\ parcial a}. Aqui, z corresponde a F_ {m - 1} (x_i) + h_m (x_i), e corresponde a F_ {m-1} (x_i).


    # A quantidade \ left [\ frac {\ partial l (y_i, F (x_i))} {\ partial F (x_i)} \ right] _ {F = F_ {m - 1}} é a derivada da perda em relação ao seu segundo parâmetro, avaliado em F_ {m-1} (x). É fácil calcular para qualquer F_ {m - 1} (x_i) dado em uma forma fechada, uma vez que a perda é diferenciável. Vamos denotá-lo por g_i


    # Removendo os termos constantes, temos:

        #  h_m \ approx \ arg \ min_ {h} \ sum_ {i = 1} ^ {n} h (x_i) g_i



    # Isso é minimizado se h (x_i) for ajustado para prever um valor que é proporcional ao gradiente negativo -g_i. Portanto, a cada iteração, o estimador h_m é ajustado para prever os gradientes negativos das amostras. Os gradientes são atualizados a cada iteração. Isso pode ser considerado como algum tipo de descida gradiente em um espaço funcional.


    # OBS: Para algumas perdas, por ex. o menor desvio absoluto (LAD) onde os gradientes são \ pm 1, os valores previstos por um h_m ajustado não são precisos o suficiente: a árvore só pode produzir valores inteiros. Como resultado, os valores das folhas da árvore h_m são modificados assim que a árvore é ajustada, de modo que os valores das folhas minimizam a perda L_m. A atualização depende da perda: para a perda LAD, o valor de uma folha é atualizado para a mediana das amostras nessa folha. 


##### 1.11.4.5.2. Classificação

    # O aumento de gradiente para classificação é muito semelhante ao caso de regressão. Porém, a soma das árvores F_M (x_i) = \ sum_m h_m (x_i) não é homogênea a uma previsão: não pode ser uma classe, pois as árvores predizem valores contínuos.

    # O mapeamento do valor F_M (x_i) para uma classe ou probabilidade depende da perda. Para o desvio (ou perda logarítmica), a probabilidade de x_i pertencer à classe positiva é modelada como p (y_i = 1 | x_i) = \ sigma (F_M (x_i)) onde \ sigma é a função sigmóide.


    # Para classificação multiclasse, árvores K (para classes K) são construídas em cada uma das M iterações. A probabilidade de x_i pertencer à classe k é modelada como um softmax dos valores F_ {M, k} (x_i).

    # Observe que mesmo para uma tarefa de classificação, o subestimador h_m ainda é um regressor, não um classificador. Isso ocorre porque os subestimadores são treinados para prever gradientes (negativos), que são sempre quantidades contínuas. 

##### 1.11.4.6. Funções de perda

    # As seguintes funções de perda são suportadas e podem ser especificadas usando o parâmetro loss:

    # Regressão:

        # Erro quadrado ('squared_error'): A escolha natural para a regressão devido às suas propriedades computacionais superiores. O modelo inicial é dado pela média dos valores alvo.

        # Mínimo desvio absoluto ('lad'): Uma função de perda robusta para regressão. O modelo inicial é dado pela mediana dos valores alvo.

        # Huber ('huber'): Outra função de perda robusta que combina mínimos quadrados e mínimo desvio absoluto; use alfa para controlar a sensibilidade em relação a outliers (consulte [F2001] para obter mais detalhes).

        # Quantil ('quantil'): uma função de perda para regressão de quantil. Use 0 <alpha <1 para especificar o quantil. Esta função de perda pode ser usada para criar intervalos de predição (consulte Intervalos de predição para regressão de aumento de gradiente).


    # Classificação

        # Desvio binomial ('desvio'): A função de perda de probabilidade logarítmica negativa binomial para classificação binária (fornece estimativas de probabilidade). O modelo inicial é dado pelo log odds ratio.

        # Desvio multinomial ('desvio'): A função de perda de probabilidade logarítmica negativa multinomial para classificação multiclasse com n_classes de classes mutuamente exclusivas. Ele fornece estimativas de probabilidade. O modelo inicial é dado pela probabilidade anterior de cada classe. Em cada iteração, árvores de regressão n_classes devem ser construídas, o que torna o GBRT bastante ineficiente para conjuntos de dados com um grande número de classes.

        # Perda exponencial ('exponencial'): A mesma função de perda que AdaBoostClassifier. Menos robusto para exemplos mal rotulados do que 'desvio'; só pode ser usado para classificação binária. 

##### 1.11.4.7. Redução por meio da taxa de aprendizagem

    # [F2001] propôs uma estratégia de regularização simples que dimensiona a contribuição de cada aluno fraco por um fator constante \ nu:


        # F_m (x) = F_ {m-1} (x) + \ nu h_m (x)


    # O parâmetro \ nu também é chamado de taxa de aprendizagem porque dimensiona o comprimento do passo do procedimento de descida do gradiente; ele pode ser definido por meio do parâmetro learning_rate.

    # O parâmetro learning_rate interage fortemente com o parâmetro n_estimators, o número de alunos fracos para ajustar. Valores menores de learning_rate requerem um número maior de alunos fracos para manter um erro de treinamento constante. A evidência empírica sugere que pequenos valores de learning_rate favorecem um melhor erro de teste. [HTF] recomenda definir a taxa de aprendizagem para uma pequena constante (por exemplo, learning_rate <= 0,1) e escolher n_estimators parando antecipadamente. Para uma discussão mais detalhada da interação entre learning_rate e n_estimators, consulte [R2007]. 



##### 1.11.4.8. Subamostragem

    # [F1999] propôs o aumento do gradiente estocástico, que combina o aumento do gradiente com a média de bootstrap (bagging). A cada iteração, o classificador base é treinado em uma subamostra de fração dos dados de treinamento disponíveis. A subamostra é desenhada sem substituição. Um valor típico de subamostra é 0,5.

    # A figura abaixo ilustra o efeito da redução e da subamostragem na adequação do modelo. Podemos ver claramente que o encolhimento supera o não encolhimento. A subamostragem com redução pode aumentar ainda mais a precisão do modelo. A subamostragem sem redução, por outro lado, tem um desempenho ruim. 

        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regularization.html

    # Outra estratégia para reduzir a variância é subamostrar os recursos análogos às divisões aleatórias em RandomForestClassifier. O número de recursos subamostrados pode ser controlado por meio do parâmetro max_features. 

    # OBS: Usar um valor pequeno de max_features pode diminuir significativamente o tempo de execução. 

    # O aumento de gradiente estocástico permite calcular estimativas out-of-bag do desvio de teste, computando a melhoria no desvio nos exemplos que não estão incluídos na amostra de bootstrap (ou seja, os exemplos out-of-bag). As melhorias são armazenadas no atributo oob_improvement_. oob_improvement_ [i] mantém a melhoria em termos de perda nas amostras OOB se você adicionar o i-ésimo estágio às previsões atuais. As estimativas out-of-bag podem ser usadas para seleção de modelo, por exemplo, para determinar o número ideal de iterações. As estimativas OOB são geralmente muito pessimistas, portanto, recomendamos usar a validação cruzada em vez disso e apenas usar OOB se a validação cruzada consumir muito tempo. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regularization.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regularization-py

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-oob-py

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html#sphx-glr-auto-examples-ensemble-plot-ensemble-oob-py


##### 1.11.4.9. Interpretação com importância do recurso 

    # As árvores de decisão individuais podem ser interpretadas facilmente, simplesmente visualizando a estrutura da árvore. Os modelos de aumento de gradiente, no entanto, compreendem centenas de árvores de regressão, portanto, não podem ser facilmente interpretados por inspeção visual das árvores individuais. Felizmente, várias técnicas foram propostas para resumir e interpretar modelos de aumento de gradiente.

    # Freqüentemente, os recursos não contribuem igualmente para prever a resposta desejada; em muitas situações, a maioria dos recursos são de fato irrelevantes. Ao interpretar um modelo, a primeira pergunta geralmente é: quais são esses recursos importantes e como eles contribuem para prever a resposta desejada?

    # As árvores de decisão individuais executam intrinsecamente a seleção de recursos, selecionando os pontos de divisão apropriados. Essas informações podem ser usadas para medir a importância de cada recurso; a ideia básica é: quanto mais frequentemente um recurso é usado nos pontos de divisão de uma árvore, mais importante é esse recurso. Essa noção de importância pode ser estendida para conjuntos de árvores de decisão simplesmente calculando a média da importância do recurso baseado em impurezas de cada árvore (consulte Avaliação da importância do recurso para obter mais detalhes).

    # As pontuações de importância do recurso de um modelo de aumento de gradiente de ajuste podem ser acessadas por meio da propriedade feature_importances_: 


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
clf.feature_importances_


    # Observe que este cálculo da importância do recurso é baseado na entropia e é diferente de sklearn.inspection.permutation_importance que é baseado na permutação dos recursos. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py