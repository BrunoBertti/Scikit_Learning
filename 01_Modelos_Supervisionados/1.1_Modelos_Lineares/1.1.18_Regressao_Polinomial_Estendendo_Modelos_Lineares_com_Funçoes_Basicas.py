########## 1.1.18 Regressão polinomial: estendendo modelos lineares com funções básicas  ##########

    # Um padrão comum no aprendizado de máquina é usar modelos lineares treinados em funções não lineares dos dados. Essa abordagem mantém o desempenho geralmente rápido dos métodos lineares, enquanto permite que eles se ajustem a uma gama muito mais ampla de dados.
    
    # Por exemplo, uma regressão linear simples pode ser estendida pela construção de recursos polinomiais a partir dos coeficientes. No caso de regressão linear padrão, você pode ter um modelo parecido com este para dados bidimensionais: 

        # Y^(w, x) = w0 + w1x1 + w2x2
    
    # If we want to fit a paraboloid to the data instead of a plane, we can combine the features in second-order polynomials, so that the model looks like this:

        # y^(w, x) = w0 + w1x1 + w2x2 + w3x1x2 + w4x1^2 + w5x2^2
    
    # A observação (às vezes surpreendente) é que este ainda é um modelo linear: para ver isso, imagine criar um novo conjunto de recursos 

        # z = [x1, x2, x1x2, x1^2, x2^2]

    # Com essa nova rotulagem dos dados, nosso problema pode ser escrito 

        # y^(w, z) = w0 + w1z1 + w2z2 + w2z3 + w4z4 + w5z5

    # Vemos que a regressão polinomial resultante está na mesma classe de modelos lineares que consideramos acima (ou seja, o modelo é linear em w) e pode ser resolvida pelas mesmas técnicas. Ao considerar ajustes lineares em um espaço de dimensão superior construído com essas funções básicas, o modelo tem a flexibilidade de se ajustar a uma gama muito mais ampla de dados. 

    # Aqui está um exemplo de aplicação dessa ideia a dados unidimensionais, usando recursos polinomiais de vários graus: 

        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html
    
    # Esta figura é criada usando o transformador PolynomialFeatures, que transforma uma matriz de dados de entrada em uma nova matriz de dados de um determinado grau. Ele pode ser usado da seguinte maneira: 

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X = np.arange(6).reshape(3,2)
print(X)
poly = PolynomialFeatures(degree=2)
print(poly.fit(X))

    # Os recursos de X foram transformados de [x1x2] para [x1, x2, x1 ^ 2, x1, x2, x2 ^ 2] e agora podem ser usados em qualquer modelo linear.

    # Esse tipo de pré-processamento pode ser simplificado com as ferramentas do Pipeline. Um único objeto que representa uma regressão polinomial simples pode ser criado e usado da seguinte maneira: 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])

# Treinar para dados de polinomio de 3 graus
x = np.arange(5)
y = 3 - 2 * x + x ** 2 - x ** 3
print(model = model.fit(x[:, np.newaxis], y))
print(model.named_steps['linear'].coef_)

    # O modelo linear treinado em características polinomiais é capaz de recuperar exatamente os coeficientes polinomiais de entrada. 

    # Em alguns casos, não é necessário incluir poderes mais elevados de qualquer recurso único, mas apenas os chamados recursos de interação que se multiplicam juntos no máximo d recursos distintos. Eles podem ser obtidos em PolynomialFeatures com a configuração interação_only = True. 

    # Por exemplo, ao lidar com recursos booleanos, xi ^ n = xi para todo n e, portanto, é inútil; mas x1xj representa a conjunção de dois booleanos. Dessa forma, podemos resolver o problema XOR com um classificador linear: 

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = X[:, 0] ^ X[:, 1]
print(y)
X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(int)
print(X)
clf = Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle=False).fit(X, y)

    # E as “previsões” do classificador são perfeitas: 


print(clf)

print(clf.score(X,y))