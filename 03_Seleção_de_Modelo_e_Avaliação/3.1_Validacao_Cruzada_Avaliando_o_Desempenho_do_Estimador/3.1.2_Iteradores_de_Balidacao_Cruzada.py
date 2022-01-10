########## 3.1.2. Iteradores de validação cruzada ##########

    # As seções a seguir listam utilitários para gerar índices que podem ser usados para gerar divisões de conjuntos de dados de acordo com diferentes estratégias de validação cruzada. 



##### 3.1.2.1. Iteradores de validação cruzada para i.i.d. dados

    # Assumir que alguns dados são independentes e distribuídos de forma idêntica (i.i.d.) é fazer a suposição de que todas as amostras derivam do mesmo processo generativo e que o processo generativo é assumido como não tendo memória de amostras geradas no passado.

    # Os seguintes validadores cruzados podem ser usados em tais casos.

    # Nota: Enquanto i.i.d. os dados são uma suposição comum na teoria do aprendizado de máquina, mas raramente se mantém na prática. Se soubermos que as amostras foram geradas usando um processo dependente do tempo, é mais seguro usar um esquema de validação cruzada ciente de série temporal. Da mesma forma, se sabemos que o processo gerador tem uma estrutura de grupo (amostras coletadas de diferentes assuntos, experimentos, dispositivos de medição), é mais seguro usar a validação cruzada do grupo. 



##### 3.1.2.1.1. K-fold

    # KFold divide todas as amostras em k grupos de amostras, chamados dobras (se k = n, isso é equivalente à estratégia Deixar de lado), de tamanhos iguais (se possível). A função de previsão é aprendida usando k - 1 dobras, e a dobra deixada de fora é usada para teste.

    # Exemplo de validação cruzada de 2 vezes em um conjunto de dados com 4 amostras: 

import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))


    # Aqui está uma visualização do comportamento da validação cruzada. Observe que o KFold não é afetado por classes ou grupos. 

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html


X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
y = np.array([0, 1, 0, 1])
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]


##### 3.1.2.1.2. K-Fold repetido


    # RepeatedKFold repete K-Fold n vezes. Pode ser usado quando é necessário executar o KFold n vezes, produzindo divisões diferentes em cada repetição.

    # Exemplo de K-Fold 2 vezes repetido 2 vezes:     

import numpy as np
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))


    # Da mesma forma, RepeatedStratifiedKFold repete Stratified K-Fold n vezes com diferentes randomizações em cada repetição. 



##### 3.1.2.1.3. Deixe um de fora (LOO)

    # LeaveOneOut (ou LOO) é uma validação cruzada simples. Cada conjunto de aprendizagem é criado retirando todas as amostras, exceto uma, sendo o conjunto de teste a amostra deixada de fora. Assim, para n amostras, temos n conjuntos de treinamento diferentes e n conjuntos de testes diferentes. Este procedimento de validação cruzada não desperdiça muitos dados, pois apenas uma amostra é removida do conjunto de treinamento: 

from sklearn.model_selection import LeaveOneOut

X = [1, 2, 3, 4]
loo = LeaveOneOut()
for train, test in loo.split(X):
    print("%s %s" % (train, test))

    # Os usuários potenciais de LOO para seleção de modelo devem considerar algumas advertências conhecidas. Quando comparado com a validação cruzada k-fold, constrói-se n modelos a partir de n amostras em vez de k modelos, onde n> k. Além disso, cada um é treinado em n - 1 amostras em vez de (k-1) n / k. De ambas as maneiras, supondo que k não seja muito grande e k <n, LOO é mais caro computacionalmente do que a validação cruzada k-fold.

    # Em termos de precisão, LOO geralmente resulta em alta variância como um estimador para o erro de teste. Intuitivamente, como n - 1 das n amostras são usadas para construir cada modelo, os modelos construídos a partir de dobras são virtualmente idênticos entre si e ao modelo construído a partir de todo o conjunto de treinamento.

    # No entanto, se a curva de aprendizado for íngreme para o tamanho do treinamento em questão, a validação cruzada de 5 ou 10 vezes pode superestimar o erro de generalização.

    # Como regra geral, a maioria dos autores e as evidências empíricas sugerem que a validação cruzada de 5 ou 10 vezes deve ser preferida a LOO. 



    ## Referências:
    ## http://www.faqs.org/faqs/ai-faq/neural-nets/part3/section-12.html;

    ## T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, Springer 2009 (https://web.stanford.edu/~hastie/ElemStatLearn/)

    ## L. Breiman, P. Spector Submodel selection and evaluation in regression: The X-random case, International Statistical Review 1992; (http://digitalassets.lib.berkeley.edu/sdtr/ucb/text/197.pdf)

    ## R. Kohavi, A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection, Intl. Jnt. Conf. AI (https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf)

    ## R. Bharat Rao, G. Fung, R. Rosales, On the Dangers of Cross-Validation. An Experimental Evaluation, SIAM 2008; (https://people.csail.mit.edu/romer/papers/CrossVal_SDM08.pdf)

    ## G. James, D. Witten, T. Hastie, R Tibshirani, An Introduction to Statistical Learning, Springer 2013. (https://www-bcf.usc.edu/~gareth/ISL/)



##### 3.1.2.1.4. Deixar P fora (LPO)


    # LeavePOut é muito semelhante a LeaveOneOut, pois cria todos os conjuntos de treinamento / teste possíveis removendo p amostras do conjunto completo. Para n amostras, isso produz {n \ escolha p} pares de teste-trem. Ao contrário de LeaveOneOut e KFold, os conjuntos de teste se sobreporão para p> 1.

    # Exemplo de Leave-2-Out em um conjunto de dados com 4 amostras:


from sklearn.model_selection import LeavePOut

X = np.ones(4)
lpo = LeavePOut(p=2)
for train, test in lpo.split(X):
    print("%s %s" % (train, test))



##### 3.1.2.1.5. Validação cruzada de permutações aleatórias a.k.a. Shuffle & Split

    # O iterador ShuffleSplit irá gerar um número definido pelo usuário de divisões de conjunto de dados de teste / treino independentes. As amostras são primeiro embaralhadas e, em seguida, divididas em um par de conjuntos de trem e de teste.

    # É possível controlar a aleatoriedade para reprodutibilidade dos resultados semeando explicitamente o gerador de números pseudo aleatórios random_state.

    # Aqui está um exemplo de uso: 

from sklearn.model_selection import ShuffleSplit
X = np.arange(10)
ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))


    # Aqui está uma visualização do comportamento da validação cruzada. Observe que ShuffleSplit não é afetado por classes ou grupos.

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html

    # ShuffleSplit é uma boa alternativa para a validação cruzada K Fold que permite um controle mais preciso sobre o número de iterações e a proporção de amostras em cada lado da divisão de trem / teste. 



##### 3.1.2.2. Iteradores de validação cruzada com estratificação baseada em rótulos de classe.

    # Alguns problemas de classificação podem exibir um grande desequilíbrio na distribuição das classes alvo: por exemplo, pode haver várias vezes mais amostras negativas do que positivas. Nesses casos, é recomendado o uso de amostragem estratificada conforme implementado em StratifiedKFold e StratifiedShuffleSplit para garantir que as frequências de classe relativas sejam preservadas aproximadamente em cada trem e dobra de validação. 



##### 3.1.2.2.1. K-fold estratificado

    # StratifiedKFold é uma variação de k-fold que retorna dobras estratificadas: cada conjunto contém aproximadamente a mesma porcentagem de amostras de cada classe de destino que o conjunto completo.

    # Aqui está um exemplo de validação cruzada estratificada de 3 vezes em um conjunto de dados com 50 amostras de duas classes não balanceadas. Mostramos o número de amostras em cada classe e comparamos com o KFold. 


from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5))
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

kf = KFold(n_splits=3)
for train, test in kf.split(X, y):
    print('train -  {}   |   test -  {}'.format(
        np.bincount(y[train]), np.bincount(y[test])))

    # Podemos ver que StratifiedKFold preserva as proporções de classe (aproximadamente 1/10) no conjunto de dados de treino e teste.

    # Aqui está uma visualização do comportamento da validação cruzada.

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html


    # RepeatedStratifiedKFold pode ser usado para repetir estratificado K-Fold n vezes com diferentes randomizações em cada repetição. 




##### 3.1.2.2.2. Divisão aleatória estratificada

    # StratifiedShuffleSplit é uma variação de ShuffleSplit, que retorna divisões estratificadas, ou seja, que cria divisões preservando a mesma porcentagem para cada classe de destino como no conjunto completo.

    # Aqui está uma visualização do comportamento da validação cruzada.

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html




##### 3.1.2.3. Iteradores de validação cruzada para dados agrupados.

    # O i.i.d. a suposição é quebrada se o processo gerador subjacente produzir grupos de amostras dependentes.

    # Esse agrupamento de dados é específico do domínio. Um exemplo seria quando há dados médicos coletados de vários pacientes, com várias amostras retiradas de cada paciente. E esses dados provavelmente dependem do grupo individual. Em nosso exemplo, a identificação do paciente para cada amostra será seu identificador de grupo.

    # Nesse caso, gostaríamos de saber se um modelo treinado em um determinado conjunto de grupos generaliza bem para os grupos invisíveis. Para medir isso, precisamos garantir que todas as amostras na dobra de validação vêm de grupos que não estão representados de forma alguma na dobra de treinamento emparelhada.

    # Os seguintes divisores de validação cruzada podem ser usados para fazer isso. O identificador de agrupamento para as amostras é especificado por meio do parâmetro groups. 


##### 3.1.2.3.1. Grupo k-fold


    # GroupKFold é uma variação de k-fold que garante que o mesmo grupo não seja representado nos conjuntos de teste e treinamento. Por exemplo, se os dados são obtidos de diferentes assuntos com várias amostras por assunto e se o modelo é flexível o suficiente para aprender com características altamente específicas da pessoa, ele pode falhar na generalização para novos assuntos. GroupKFold torna possível detectar este tipo de situações de overfitting.

    # Imagine que você tem três assuntos, cada um com um número associado de 1 a 3: 

from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))

    # Cada sujeito está em uma dobra de teste diferente, e o mesmo sujeito nunca está tanto no teste quanto no treinamento. Observe que as dobras não têm exatamente o mesmo tamanho devido ao desequilíbrio nos dados.

    # Aqui está uma visualização do comportamento da validação cruzada. 


        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html




##### 3.1.2.3.2. StratifiedGroupKFold

    # StratifiedGroupKFold é um esquema de validação cruzada que combina StratifiedKFold e GroupKFold. A ideia é tentar preservar a distribuição das classes em cada divisão, mantendo cada grupo dentro de uma única divisão. Isso pode ser útil quando você tem um conjunto de dados não balanceado, de forma que usar apenas GroupKFold pode produzir divisões distorcidas.

    # Exemplo: 
from sklearn.model_selection import StratifiedGroupKFold
X = list(range(18))
y = [1] * 6 + [0] * 12
groups = [1, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6]
sgkf = StratifiedGroupKFold(n_splits=3)
for train, test in sgkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))


    # Notas de implementação: 

        # Com a implementação atual, o embaralhamento completo não é possível na maioria dos cenários. Quando shuffle = True, acontece o seguinte: 

            # 1 - Todos os grupos são embaralhados.

            # 2 - Os grupos são classificados pelo desvio padrão das classes usando classificação estável.

            # 3 - Os grupos classificados são iterados e atribuídos a dobras.

            # Isso significa que apenas grupos com o mesmo desvio padrão de distribuição de classe serão embaralhados, o que pode ser útil quando cada grupo tem apenas uma classe. 

        # O algoritmo atribui avidamente cada grupo a um dos conjuntos de teste n_splits, escolhendo o conjunto de teste que minimiza a variação na distribuição de classe entre os conjuntos de teste. A atribuição de grupo procede de grupos com variação mais alta para mais baixa na frequência de classe, ou seja, grupos grandes com pico em uma ou poucas classes são atribuídos primeiro.

        # Essa divisão é subótima no sentido de que pode produzir divisões desequilibradas, mesmo se a estratificação perfeita for possível. Se você tiver uma distribuição relativamente próxima das classes em cada grupo, usar GroupKFold é melhor 


    # Aqui está uma visualização do comportamento de validação cruzada para grupos desiguais: 

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html




##### 3.1.2.3.3. Deixe um grupo de fora

    # LeaveOneGroupOut é um esquema de validação cruzada que mantém as amostras de acordo com uma matriz de grupos de inteiros fornecida por terceiros. Essas informações de grupo podem ser usadas para codificar dobras de validação cruzada predefinidas de domínio arbitrário específico.

    # Cada conjunto de treinamento é assim constituído por todas as amostras, exceto aquelas relacionadas a um grupo específico.

    # Por exemplo, nos casos de vários experimentos, LeaveOneGroupOut pode ser usado para criar uma validação cruzada com base nos diferentes experimentos: criamos um conjunto de treinamento usando as amostras de todos os experimentos, exceto um: 


from sklearn.model_selection import LeaveOneGroupOut

X = [1, 5, 10, 50, 60, 70, 80]
y = [0, 1, 1, 2, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3, 3]
logo = LeaveOneGroupOut()
for train, test in logo.split(X, y, groups=groups):
    print("%s %s" % (train, test))


    # Outra aplicação comum é usar informações de tempo: por exemplo, os grupos podem ser o ano de coleta das amostras e, assim, permitir a validação cruzada em relação às divisões baseadas no tempo.


##### 3.1.2.3.4. Deixar P Grupos de Fora

    # LeavePGroupsOut é semelhante a LeaveOneGroupOut, mas remove amostras relacionadas a grupos P para cada conjunto de treinamento / teste.

    # Exemplo de Leave-2-Group Out: 

from sklearn.model_selection import LeavePGroupsOut

X = np.arange(6)
y = [1, 1, 1, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3]
lpgo = LeavePGroupsOut(n_groups=2)
for train, test in lpgo.split(X, y, groups=groups):
    print("%s %s" % (train, test))

##### 3.1.2.3.5. Divisão de grupo aleatório

    # O iterador GroupShuffleSplit se comporta como uma combinação de ShuffleSplit e LeavePGroupsOut e gera uma sequência de partições aleatórias em que um subconjunto de grupos é mantido para cada divisão.

    # Aqui está um exemplo de uso: 

from sklearn.model_selection import GroupShuffleSplit

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("%s %s" % (train, test))


    # Aqui está uma visualização do comportamento da validação cruzada. 

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html

    # Esta classe é útil quando o comportamento de LeavePGroupsOut é desejado, mas o número de grupos é grande o suficiente para que gerar todas as partições possíveis com grupos P retidos seria proibitivamente caro. Nesse cenário, GroupShuffleSplit fornece uma amostra aleatória (com substituição) das divisões de trem / teste geradas por LeavePGroupsOut. 

##### 3.1.2.4. Conjuntos de Dobras / Validação Predefinidos

    # Para alguns conjuntos de dados, já existe uma divisão predefinida dos dados em dobra de treinamento e validação ou em várias dobras de validação cruzada. Usando PredefinedSplit, é possível usar essas dobras, por exemplo ao pesquisar hiperparâmetros.

    # Por exemplo, ao usar um conjunto de validação, defina test_fold como 0 para todas as amostras que fazem parte do conjunto de validação e como -1 para todas as outras amostras. 



##### 3.1.2.5. Usando iteradores de validação cruzada para dividir o treinamento e o teste

    # As funções de validação cruzada do grupo acima também podem ser úteis para dividir um conjunto de dados em subconjuntos de treinamento e teste. Observe que a função de conveniência train_test_split é um invólucro em torno de ShuffleSplit e, portanto, permite apenas a divisão estratificada (usando os rótulos de classe) e não pode contabilizar grupos.

    # Para executar a divisão de trem e teste, use os índices para o trem e subconjuntos de teste produzidos pela saída do gerador pelo método split () do divisor de validação cruzada. Por exemplo:

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

X = np.array([0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001])
y = np.array(["a", "b", "b", "b", "c", "c", "c", "a"])
groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
train_indx, test_indx = next(
    GroupShuffleSplit(random_state=7).split(X, y, groups)
)
X_train, X_test, y_train, y_test = \
    X[train_indx], X[test_indx], y[train_indx], y[test_indx]
X_train.shape, X_test.shape

np.unique(groups[train_indx]), np.unique(groups[test_indx])



##### 3.1.2.6. Validação cruzada de dados de série temporal

    # Os dados de série temporal são caracterizados pela correlação entre observações que estão próximas no tempo (autocorrelação). No entanto, as técnicas clássicas de validação cruzada, como KFold e ShuffleSplit, presumem que as amostras são independentes e distribuídas de forma idêntica e resultariam em uma correlação irracional entre as instâncias de treinamento e teste (gerando estimativas ruins de erro de generalização) em dados de série temporal. Portanto, é muito importante avaliar nosso modelo para dados de série temporal nas observações “futuras”, pelo menos como aquelas que são usadas para treinar o modelo. Para conseguir isso, uma solução é fornecida pela TimeSeriesSplit. 


##### 3.1.2.6.1. Divisão de série temporal 

    # TimeSeriesSplit é uma variação de k-fold que retorna as primeiras k dobras como conjunto de trem e a (k + 1) ésima dobra como conjunto de teste. Observe que, ao contrário dos métodos de validação cruzada padrão, conjuntos de treinamento sucessivos são superconjuntos daqueles que vêm antes deles. Além disso, ele adiciona todos os dados excedentes à primeira partição de treinamento, que sempre é usada para treinar o modelo.

    # Esta classe pode ser usada para validação cruzada de amostras de dados de série temporal que são observadas em intervalos de tempo fixos.

    # Exemplo de validação cruzada de série temporal de 3 divisões em um conjunto de dados com 6 amostras: 

from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)
TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None)
for train, test in tscv.split(X):
    print("%s %s" % (train, test))

    # Aqui está uma visualização do comportamento da validação cruzada. 

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html

        