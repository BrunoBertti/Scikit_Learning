########## 1.14. Aprendizagem Semissupervisionada ##########

    # A aprendizagem semissupervisionada é uma situação em que, nos dados de treinamento, algumas das amostras não são rotuladas. Os estimadores semissupervisionados em sklearn.semi_supervised são capazes de fazer uso desses dados não rotulados adicionais para capturar melhor a forma da distribuição de dados subjacente e generalizar melhor para novas amostras. Esses algoritmos podem funcionar bem quando temos uma quantidade muito pequena de pontos rotulados e uma grande quantidade de pontos não rotulados.

    # Entradas não marcadas em y

        # É importante atribuir um identificador a pontos não rotulados junto com os dados rotulados ao treinar o modelo com o método de ajuste. O identificador que esta implementação usa é o valor inteiro. Observe que, para rótulos de string, o dtype de y deve ser objeto de forma que possa conter strings e inteiros.

    # Observação: algoritmos semissupervisionados precisam fazer suposições sobre a distribuição do conjunto de dados para obter ganhos de desempenho. Veja aqui para mais detalhes. 