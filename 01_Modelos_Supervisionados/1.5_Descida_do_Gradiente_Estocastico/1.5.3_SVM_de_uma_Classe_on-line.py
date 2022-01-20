########## 1.5.3. SVM de uma classe on-line ##########


    # A classe sklearn.linear_model.SGDOneClassSVM implementa uma versão linear online do One-Class SVM usando uma descida de gradiente estocástica. Combinado com técnicas de aproximação de kernel, o sklearn.linear_model.SGDOneClassSVM pode ser usado para aproximar a solução de um SVM One-Class kernelizado, implementado em sklearn.svm.OneClassSVM, com uma complexidade linear no número de amostras. Observe que a complexidade de um SVM de uma classe com kernel é, na melhor das hipóteses, quadrática no número de amostras. sklearn.linear_model.SGDOneClassSVM é, portanto, adequado para conjuntos de dados com um grande número de amostras de treinamento (> 10.000) para os quais a variante SGD pode ser várias ordens de magnitude mais rápida.

    # Sua implementação é baseada na implementação do gradiente descendente estocástico. De fato, o problema de otimização original do SVM de uma classe é dado por 

        # \begin{split}\begin{aligned}
        # \min_{w, \rho, \xi} & \quad \frac{1}{2}\Vert w \Vert^2 - \rho + \frac{1}{\nu n} \sum_{i=1}^n \xi_i \\
        # \text{s.t.} & \quad \langle w, x_i \rangle \geq \rho - \xi_i \quad 1 \leq i \leq n \\
        # & \quad \xi_i \geq 0 \quad 1 \leq i \leq n
        # \end{aligned}\end{split}


    # onde \nu \in (0, 1] é o parâmetro especificado pelo usuário que controla a proporção de outliers e a proporção de vetores de suporte. Livrar-se das variáveis de folga \xi_i este problema é equivalente a 

        # \min_{w, \rho} \frac{1}{2}\Vert w \Vert^2 - \rho + \frac{1}{\nu n} \sum_{i=1}^n \max(0, \rho - \langle w, x_i \rangle) \, .

    # Multiplicando pela constante \nu e introduzindo o intercepto b = 1 - \rho obtemos o seguinte problema de otimização equivalente 

        # \min_{w, b} \frac{\nu}{2}\Vert w \Vert^2 + b\nu + \frac{1}{n} \sum_{i=1}^n \max(0, 1 - (\langle w, x_i \rangle + b)) \, .


    # Isto é similar aos problemas de otimização estudados na seção Formulação matemática com y_i = 1, 1 \leq i \leq n e \alpha = \nu/2, sendo a função de perda de dobradiça e R sendo a norma L2. Só precisamos adicionar o termo b\nu no loop de otimização.

    # Como SGDClassifier e SGDRegressor, SGDOneClassSVM suporta SGD médio. A média pode ser habilitada definindo average=True. 