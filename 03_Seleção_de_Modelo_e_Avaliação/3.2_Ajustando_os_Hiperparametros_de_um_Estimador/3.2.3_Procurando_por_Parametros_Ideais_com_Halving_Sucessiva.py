########## 3.2.3. Procurando por parâmetros ideais com halving sucessiva ##########

    # O Scikit-learn também fornece os estimadores HalvingGridSearchCV e HalvingRandomSearchCV que podem ser usados ​​para pesquisar um espaço de parâmetros usando halvings sucessivos 1 2. O halving sucessivo (SH) é como um torneio entre combinações de parâmetros candidatos. SH é um processo de seleção iterativo onde todos os candidatos (as combinações de parâmetros) são avaliados com uma pequena quantidade de recursos na primeira iteração. Apenas alguns desses candidatos são selecionados para a próxima iteração, à qual serão alocados mais recursos. Para ajuste de parâmetro, o recurso normalmente é o número de amostras de treinamento, mas também pode ser um parâmetro numérico arbitrário, como n_estimators em uma floresta aleatória.

    # Conforme ilustrado na figura abaixo, apenas um subconjunto de candidatos “sobrevive” até a última iteração. Esses são os candidatos que se classificaram consistentemente entre os candidatos com melhor pontuação em todas as iterações. A cada iteração é alocada uma quantidade crescente de recursos por candidato, aqui o número de amostras. 


        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_successive_halving_iterations.html


    # Descrevemos aqui brevemente os principais parâmetros, mas cada parâmetro e suas interações são descritos com mais detalhes nas seções abaixo. O parâmetro fator (> 1) controla a taxa de crescimento dos recursos e a taxa de redução do número de candidatos. Em cada iteração, o número de recursos por candidato é multiplicado por fator e o número de candidatos é dividido pelo mesmo fator. Junto com resource e min_resources, factor é o parâmetro mais importante para controlar a busca em nossa implementação, embora um valor de 3 geralmente funcione bem. factor controla efetivamente o número de iterações em HalvingGridSearchCV e o número de candidatos (por padrão) e iterações em HalvingRandomSearchCV. agressivo_elimination=True também pode ser usado se o número de recursos disponíveis for pequeno. Mais controle está disponível por meio do ajuste do parâmetro min_resources.

    # Esses estimadores ainda são experimentais: suas previsões e sua API podem mudar sem qualquer ciclo de depreciação. Para usá-los, você precisa importar explicitamente enable_halving_search_cv: 

# exigir explicitamente esse recurso experimental 
from sklearn.experimental import enable_halving_search_cv  # noqa (SEM Garantia de Qualidade)
# agora você pode importar normalmente de model_selection 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/model_selection/plot_successive_halving_heatmap.html#sphx-glr-auto-examples-model-selection-plot-successive-halving-heatmap-py

    ## https://scikit-learn.org/stable/auto_examples/model_selection/plot_successive_halving_iterations.html#sphx-glr-auto-examples-model-selection-plot-successive-halving-iterations-py


##### 3.2.3.1. Escolhendo min_resources e o número de candidatos

    # Além do fator, os dois principais parâmetros que influenciam o comportamento de uma busca sucessiva pela metade são o parâmetro min_resources e o número de candidatos (ou combinações de parâmetros) que são avaliados. min_resources é a quantidade de recursos alocados na primeira iteração para cada candidato. O número de candidatos é especificado diretamente em HalvingRandomSearchCV e é determinado a partir do parâmetro param_grid de HalvingGridSearchCV.

    # Considere um caso em que o recurso é o número de amostras e onde temos 1.000 amostras. Em teoria, com min_resources=10 e factor=2, podemos executar no máximo 7 iterações com o seguinte número de amostras: [10, 20, 40, 80, 160, 320, 640].

    # Mas dependendo do número de candidatos, podemos executar menos de 7 iterações: se começarmos com um número pequeno de candidatos, a última iteração pode usar menos de 640 amostras, o que significa não usar todos os recursos disponíveis (amostras). Por exemplo, se começarmos com 5 candidatos, precisamos apenas de 2 iterações: 5 candidatos para a primeira iteração, depois 5 // 2 = 2 candidatos na segunda iteração, após o que sabemos qual candidato tem o melhor desempenho (portanto, não preciso de um terceiro). Estaríamos usando no máximo 20 amostras, o que é um desperdício, pois temos 1000 amostras à nossa disposição. Por outro lado, se começarmos com um número alto de candidatos, podemos acabar com muitos candidatos na última iteração, o que nem sempre é o ideal: significa que muitos candidatos concorrerão com todos os recursos, basicamente reduzindo o procedimento para pesquisa padrão.

    # No caso de HalvingRandomSearchCV, o número de candidatos é definido por padrão de forma que a última iteração use o máximo possível dos recursos disponíveis. Para HalvingGridSearchCV, o número de candidatos é determinado pelo parâmetro param_grid. Alterar o valor de min_resources afetará o número de iterações possíveis e, como resultado, também afetará o número ideal de candidatos.

    # Outra consideração ao escolher min_resources é se é ou não fácil discriminar entre bons e maus candidatos com uma pequena quantidade de recursos. Por exemplo, se você precisar de muitas amostras para distinguir entre parâmetros bons e ruins, um valor alto de min_resources é recomendado. Por outro lado, se a distinção for clara mesmo com uma pequena quantidade de amostras, um pequeno min_resources pode ser preferível, pois aceleraria o cálculo.

    # Observe no exemplo acima que a última iteração não usa a quantidade máxima de recursos disponíveis: 1000 amostras estão disponíveis, mas apenas 640 são usadas, no máximo. Por padrão, tanto HalvingRandomSearchCV quanto HalvingGridSearchCV tentam usar tantos recursos quanto possível na última iteração, com a restrição de que essa quantidade de recursos deve ser um múltiplo de min_resources e fator (essa restrição será clara na próxima seção). HalvingRandomSearchCV consegue isso amostrando a quantidade certa de candidatos, enquanto HalvingGridSearchCV consegue isso configurando corretamente min_resources. Consulte Esgotando os recursos disponíveis para obter detalhes. 

##### 3.2.3.2. Quantidade de recursos e número de candidatos em cada iteração

    # Em qualquer iteração i, cada candidato recebe uma determinada quantidade de recursos que denotamos n_resources_i. Esta quantidade é controlada pelos parâmetros factor e min_resources da seguinte forma (factor é estritamente superior a 1):

        # n_resources_i = fator**i * min_resources,

    # ou equivalente:

        # n_resources_{i+1} = n_resources_i * fator

    # onde min_resources == n_resources_0 é a quantidade de recursos usados ​​na primeira iteração. fator também define as proporções de candidatos que serão selecionados para a próxima iteração:

        # n_candidates_i = n_candidates // (fator ** i)

    # ou equivalente:

        # n_candidatos_0 = n_candidatos
        # n_candidates_{i+1} = n_candidates_i // fator

    # Então, na primeira iteração, usamos recursos min_resources n_candidates times. Na segunda iteração, usamos min_resources * fator recursos n_candidates // fator vezes. O terceiro novamente multiplica os recursos por candidato e divide o número de candidatos. Esse processo é interrompido quando a quantidade máxima de recursos por candidato é atingida ou quando identificamos o melhor candidato. O melhor candidato é identificado na iteração que está avaliando o fator ou menos candidatos (veja logo abaixo uma explicação).

    # Aqui está um exemplo com min_resources=3 e fator=2, começando com 70 candidatos: 

 # n_resources_i                n_candidates_i
 # 3 (=min_resources)           70 (=n_candidates)
 # 3 * 2 = 6                    70 // 2 = 35
 # 6 * 2 = 12                   35 // 2 = 17
 # 12 * 2 = 24                  17 // 2 = 8
 # 24 * 2 = 48                  8 // 2 = 4
 # 48 * 2 = 96                  4 // 2 = 2


    # Podemos notar que: 

        # o processo para na primeira iteração que avalia fator=2 candidatos: o melhor candidato é o melhor desses 2 candidatos. Não é necessário executar uma iteração adicional, pois avaliaria apenas um candidato (ou seja, o melhor, que já identificamos). Por esse motivo, em geral, queremos que a última iteração seja executada na maioria dos candidatos a fatores. Se a última iteração avaliar mais do que candidatos a fatores, essa última iteração se reduzirá a uma pesquisa regular (como em RandomizedSearchCV ou GridSearchCV).

        # cada n_resources_i é um múltiplo de fator e min_resources (o que é confirmado por sua definição acima).

    # A quantidade de recursos que é usada em cada iteração pode ser encontrada no atributo n_resources_. 


##### 3.2.3.3. Escolhendo um recurso

    # Por padrão, o recurso é definido em termos de número de amostras. Ou seja, cada iteração usará uma quantidade crescente de amostras para treinar. No entanto, você pode especificar manualmente um parâmetro a ser usado como recurso com o parâmetro resource. Aqui está um exemplo onde o recurso é definido em termos do número de estimadores de uma floresta aleatória:

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd
param_grid = {'max_depth': [3, 5, 10],
              'min_samples_split': [2, 5, 10]}
base_estimator = RandomForestClassifier(random_state=0)
X, y = make_classification(n_samples=1000, random_state=0)
sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=2, resource='n_estimators',
                         max_resources=30).fit(X, y)
sh.best_estimator_




    # Observe que não é possível orçar em um parâmetro que faz parte da grade de parâmetros. 


##### 3.2.3.4. Esgotar os recursos disponíveis

    # Conforme mencionado acima, o número de recursos usados em cada iteração depende do parâmetro min_resources. Se você tiver muitos recursos disponíveis, mas começar com um número baixo de recursos, alguns deles podem ser desperdiçados (ou seja, não usados): 


from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd
param_grid= {'kernel': ('linear', 'rbf'),
             'C': [1, 10, 100]}
base_estimator = SVC(gamma='scale')
X, y = make_classification(n_samples=1000)
sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=2, min_resources=20).fit(X, y)
sh.n_resources_


    # O processo de pesquisa usará apenas 80 recursos no máximo, enquanto nossa quantidade máxima de recursos disponíveis é n_samples=1000. Aqui, temos min_resources = r_0 = 20. 

    # Para HalvingGridSearchCV, por padrão, o parâmetro min_resources é definido como ‘exhaust’. Isso significa que min_resources é definido automaticamente de forma que a última iteração possa usar tantos recursos quanto possível, dentro do limite max_resources: 

sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=2, min_resources='exhaust').fit(X, y)
sh.n_resources_

    # min_resources foi aqui automaticamente definido como 250, o que resulta na última iteração usando todos os recursos. O valor exato que é usado depende do número de parâmetro candidato, em max_resources e no fator.

    # Para HalvingRandomSearchCV, esgotar os recursos pode ser feito de 2 maneiras: 

        # definindo min_resources='exhaust', assim como para HalvingGridSearchCV;

        # definindo n_candidates='exaustão'. 

    # Ambas as opções são mutuamente exclusivas: usar min_resources='exhaust' requer conhecer o número de candidatos, e simetricamente n_candidates='exhaust' requer conhecer min_resources.

    # Em geral, esgotar o número total de recursos leva a um melhor parâmetro de candidato final e é um pouco mais demorado. 


##### 3.2.3.5. Eliminação agressiva de candidatos

    # Idealmente, queremos que a última iteração avalie os candidatos a fatores (consulte Quantidade de recursos e número de candidatos em cada iteração). Depois é só escolher o melhor. Quando o número de recursos disponíveis é pequeno em relação ao número de candidatos, a última iteração pode ter que avaliar mais do que candidatos a fator: 

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd

param_grid = {'kernel': ('linear', 'rbf'),
              'C': [1, 10, 100]}
base_estimator = SVC(gamma='scale')
X, y = make_classification(n_samples=1000)
sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=2, max_resources=40,
                         aggressive_elimination=False).fit(X, y)

sh.n_resources_

sh.n_candidates_

    # Como não podemos usar mais de max_resources=40 recursos, o processo deve parar na segunda iteração que avalia mais de fator=2 candidatos.

    # Usando o parâmetro agressivo_eliminação, você pode forçar o processo de pesquisa a terminar com menos de candidatos a fator na última iteração. Para isso, o processo eliminará quantos candidatos forem necessários usando os recursos min_resources: 

sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                           factor=2,
                           max_resources=40,
                           aggressive_elimination=True,
                           ).fit(X, y)
sh.n_resources_

sh.n_candidates_

    # Observe que terminamos com 2 candidatos na última iteração, pois eliminamos candidatos suficientes durante as primeiras iterações, usando n_resources = min_resources = 20. 





##### 3.2.3.6. Analisando resultados com o atributo cv_results_ 

    # O atributo cv_results_ contém informações úteis para analisar os resultados de uma pesquisa. Ele pode ser convertido em um dataframe pandas com df = pd.DataFrame(est.cv_results_). O atributo cv_results_ de HalvingGridSearchCV e HalvingRandomSearchCV é semelhante ao de GridSearchCV e RandomizedSearchCV, com informações adicionais relacionadas ao processo de halving sucessivo.

    # Aqui está um exemplo com algumas das colunas de um dataframe (truncado): 



                                #########################
                                #####    TABELA     #####
                                #####    TABELA     #####
                                #########################


    # Cada linha corresponde a uma determinada combinação de parâmetros (um candidato) e uma determinada iteração. A iteração é dada pela coluna iter. A coluna n_resources informa quantos recursos foram usados.

    # No exemplo acima, a melhor combinação de parâmetros é {'criterion': 'entropy', 'max_depth': None, 'max_features': 9, 'min_samples_split': 10} desde que atingiu a última iteração (3) com a maior pontuação: 0,96. 



    ## Referências:

    ## K. Jamieson, A. Talwalkar, Non-stochastic Best Arm Identification and Hyperparameter Optimization, in proc. of Machine Learning Research, 2016. (http://proceedings.mlr.press/v51/jamieson16.html)

    ## L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, A. Talwalkar, Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization, in Machine Learning Research 18, 2018. (https://arxiv.org/abs/1603.06560)