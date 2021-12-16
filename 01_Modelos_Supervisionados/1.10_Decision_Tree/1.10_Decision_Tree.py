########## 1.10 Decision Tree ##########

    # Árvores de decisão (DTs) são um método de aprendizado supervisionado não paramétrico usado para classificação e regressão. O objetivo é criar um modelo que preveja o valor de uma variável de destino, aprendendo regras de decisão simples inferidas dos recursos de dados. Uma árvore pode ser vista como uma aproximação constante por partes.

    # Por exemplo, no exemplo abaixo, as árvores de decisão aprendem com os dados a aproximar uma curva senoidal com um conjunto de regras de decisão if-then-else. Quanto mais profunda a árvore, mais complexas são as regras de decisão e mais adequado é o modelo. 

        # https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

    # Algumas vantagens das árvores de decisão são:

        # Simples de entender e interpretar. As árvores podem ser visualizadas.

        # Requer pouca preparação de dados. Outras técnicas geralmente requerem normalização de dados, variáveis ​​fictícias precisam ser criadas e valores em branco devem ser removidos. Observe, entretanto, que este módulo não oferece suporte a valores ausentes.

        # O custo de usar a árvore (ou seja, dados de previsão) é logarítmico no número de pontos de dados usados ​​para treinar a árvore.

        # Capaz de lidar com dados numéricos e categóricos. No entanto, a implementação do scikit-learn não oferece suporte a variáveis ​​categóricas por enquanto. Outras técnicas geralmente são especializadas na análise de conjuntos de dados que possuem apenas um tipo de variável. Veja algoritmos para mais informações.

        # Capaz de lidar com problemas de múltiplas saídas.

        # Usa um modelo de caixa branca. Se uma determinada situação é observável em um modelo, a explicação para a condição é facilmente explicada pela lógica booleana. Por outro lado, em um modelo de caixa preta (por exemplo, em uma rede neural artificial), os resultados podem ser mais difíceis de interpretar.

        # Possível validar um modelo por meio de testes estatísticos. Isso torna possível contabilizar a confiabilidade do modelo.

        # Apresenta um bom desempenho, mesmo que suas suposições sejam violadas de alguma forma pelo modelo verdadeiro a partir do qual os dados foram gerados. 

    # As desvantagens das árvores de decisão incluem:

        # Os alunos da árvore de decisão podem criar árvores excessivamente complexas que não generalizam bem os dados. Isso é chamado de overfitting. Mecanismos como poda, definir o número mínimo de amostras necessárias em um nó folha ou definir a profundidade máxima da árvore são necessários para evitar este problema.

        # As árvores de decisão podem ser instáveis ​​porque pequenas variações nos dados podem resultar na geração de uma árvore completamente diferente. Esse problema é atenuado pelo uso de árvores de decisão em um conjunto.

        # As previsões das árvores de decisão não são suaves nem contínuas, mas aproximações constantes por partes, conforme visto na figura acima. Portanto, eles não são bons em extrapolação.

        # O problema de aprender uma árvore de decisão ótima é conhecido por ser NP-completo sob vários aspectos de otimização e até mesmo para conceitos simples. Consequentemente, algoritmos de aprendizagem de árvore de decisão práticos são baseados em algoritmos heurísticos, como o algoritmo guloso, em que decisões localmente ótimas são feitas em cada nó. Esses algoritmos não podem garantir o retorno da árvore de decisão globalmente ótima. Isso pode ser mitigado pelo treinamento de várias árvores em um aluno de conjunto, onde os recursos e as amostras são amostrados aleatoriamente com substituição.

        # Existem conceitos que são difíceis de aprender porque as árvores de decisão não os expressam facilmente, como problemas de XOR, paridade ou multiplexador.

        # Os alunos da árvore de decisão criam árvores tendenciosas se algumas classes dominam. Portanto, é recomendável equilibrar o conjunto de dados antes de ajustá-lo à árvore de decisão. 