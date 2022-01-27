########## 4.1.4. Métodos de computação ##########


    # Existem dois métodos principais para aproximar a integral acima, ou seja, os métodos 'bruto' e 'recursivo'. O parâmetro method controla qual método usar.

    # O método ‘bruto’ é um método genérico que funciona com qualquer estimador. Observe que a computação de gráficos ICE é suportada apenas com o método 'bruto'. Ele aproxima a integral acima calculando uma média sobre os dados X:

        # pd_{X_S}(x_S) \approx \frac{1}{n_\text{amostras}} \sum_{i=1}^n f(x_S, x_C^{(i)}),

    # onde x_C^{(i)} é o valor da i-ésima amostra para os recursos em X_C. Para cada valor de x_S, este método requer uma passagem completa sobre o conjunto de dados X que é computacionalmente intensivo.

    # Cada um dos f(x_{S}, x_{C}^{(i)}) corresponde a uma linha ICE avaliada em x_{S}. Calculando isso para vários valores de x_{S}, obtém-se uma linha ICE completa. Como se pode ver, a média das linhas do ICE corresponde à linha de dependência parcial.

    # O método 'recursão' é mais rápido que o método 'bruto', mas é suportado apenas para gráficos PDP por alguns estimadores baseados em árvores. É calculado da seguinte forma. Para um dado ponto x_S, uma travessia de árvore ponderada é executada: se um nó de divisão envolve um recurso de entrada de interesse, o ramo esquerdo ou direito correspondente é seguido; caso contrário, ambas as ramificações são seguidas, cada ramificação sendo ponderada pela fração de amostras de treinamento que entraram nessa ramificação. Finalmente, a dependência parcial é dada por uma média ponderada de todos os valores das folhas visitadas.

    # Com o método ‘brute’, o parâmetro X é usado tanto para gerar a grade de valores x_S quanto para os valores de recurso de complemento x_C. No entanto, com o método 'recursão', X é usado apenas para os valores da grade: implicitamente, os valores x_C são os dos dados de treinamento.

    # Por padrão, o método 'recursão' é usado para plotar PDPs em estimadores baseados em árvore que o suportam, e 'bruto' é usado para o resto.


    # Nota: Embora ambos os métodos devam ser próximos em geral, eles podem diferir em algumas configurações específicas. O método ‘brute’ assume a existência dos pontos de dados (x_S, x_C^{(i)}). Quando as características são correlacionadas, tais amostras artificiais podem ter uma massa de probabilidade muito baixa. Os métodos “bruto” e “recursivo” provavelmente discordarão em relação ao valor da dependência parcial, porque tratarão essas amostras improváveis ​​de maneira diferente. Lembre-se, no entanto, que a principal suposição para interpretar PDPs é que os recursos devem ser independentes. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py





    ## Notas de rodapé

    ## 1 Para classificação, a resposta alvo pode ser a probabilidade de uma classe (a classe positiva para classificação binária) ou a função de decisão. 



    ## Referências:

    ## T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical Learning, Second Edition, Section 10.13.2, Springer, 2009. (https://web.stanford.edu/~hastie/ElemStatLearn//)

    ## C. Molnar, Interpretable Machine Learning, Section 5.1, 2019. (https://christophm.github.io/interpretable-ml-book/)

    ## A. Goldstein, A. Kapelner, J. Bleich, and E. Pitkin, Peeking Inside the Black Box: Visualizing Statistical Learning With Plots of Individual Conditional Expectation, Journal of Computational and Graphical Statistics, 24(1): 44-65, Springer, 2015. (https://arxiv.org/abs/1309.6392)