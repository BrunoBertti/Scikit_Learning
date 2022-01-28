########## 4.2. Importância do recurso de permutação ##########

    # A importância do recurso de permutação é uma técnica de inspeção de modelo que pode ser usada para qualquer estimador ajustado quando os dados são tabulares. Isso é especialmente útil para estimadores não lineares ou opacos. A importância do recurso de permutação é definida como a diminuição na pontuação de um modelo quando um único valor de recurso é embaralhado aleatoriamente 1. Este procedimento quebra a relação entre o recurso e o destino, portanto, a queda na pontuação do modelo é indicativa de quanto o modelo depende do recurso. Essa técnica se beneficia de ser agnóstica do modelo e pode ser calculada muitas vezes com diferentes permutações do recurso.

    # viso: os recursos considerados de baixa importância para um modelo ruim (baixa pontuação de validação cruzada) podem ser muito importantes para um bom modelo. Portanto, é sempre importante avaliar o poder preditivo de um modelo usando um conjunto retido (ou melhor com validação cruzada) antes de calcular as importâncias. A importância da permutação não reflete o valor preditivo intrínseco de um recurso por si só, mas a importância desse recurso para um modelo específico.
    # A função permutation_importance calcula a importância do recurso de estimadores para um determinado conjunto de dados. O parâmetro n_repeats define o número de vezes que um recurso é embaralhado aleatoriamente e retorna uma amostra de importâncias do recurso.

    # Vamos considerar o seguinte modelo de regressão treinado: 


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
diabetes = load_diabetes()
X_train, X_val, y_train, y_val = train_test_split(
    diabetes.data, diabetes.target, random_state=0)

model = Ridge(alpha=1e-2).fit(X_train, y_train)
model.score(X_val, y_val)

    # Seu desempenho de validação, medido por meio da pontuação R^2, é significativamente maior do que o nível de chance. Isso torna possível usar a função permutation_importance para sondar quais recursos são mais preditivos: 

from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_val, y_val,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{diabetes.feature_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")


    # Observe que os valores de importância para os principais recursos representam uma grande fração da pontuação de referência de 0,356.

    # As importâncias de permutação podem ser calculadas no conjunto de treinamento ou em um conjunto de teste ou validação retido. O uso de um conjunto retido permite destacar quais recursos mais contribuem para o poder de generalização do modelo inspecionado. Os recursos que são importantes no conjunto de treinamento, mas não no conjunto retido, podem fazer com que o modelo se ajuste demais.

    # A importância do recurso de permutação é a diminuição na pontuação de um modelo quando um único valor de recurso é embaralhado aleatoriamente. A função de pontuação a ser usada para o cálculo de importâncias pode ser especificada com o argumento de pontuação, que também aceita vários pontuadores. Usar vários pontuadores é mais eficiente computacionalmente do que chamar sequencialmente permutation_importance várias vezes com um pontuador diferente, pois reutiliza as previsões do modelo.

    # Um exemplo de uso de vários pontuadores é mostrado abaixo, empregando uma lista de métricas, mas mais formatos de entrada são possíveis, conforme documentado em Uso de avaliação de várias métricas. 

scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
r_multi = permutation_importance(
    model, X_val, y_val, n_repeats=30, random_state=0, scoring=scoring)

for metric in r_multi:
    print(f"{metric}")
    r = r_multi[metric]
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"    {diabetes.feature_names[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

    # A classificação dos recursos é aproximadamente a mesma para diferentes métricas, mesmo que as escalas dos valores de importância sejam muito diferentes. No entanto, isso não é garantido e métricas diferentes podem levar a importâncias de recursos significativamente diferentes, em particular para modelos treinados para problemas de classificação desequilibrada, para os quais a escolha da métrica de classificação pode ser crítica. 