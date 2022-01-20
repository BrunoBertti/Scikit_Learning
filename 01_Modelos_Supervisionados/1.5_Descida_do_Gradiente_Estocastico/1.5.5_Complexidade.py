########## 1.5.5. Complexidade ##########

    # A grande vantagem do SGD é sua eficiência, que é basicamente linear no número de exemplos de treinamento. Se X é uma matriz de tamanho (n, p) o treinamento tem um custo de O(kn \bar p), onde k é o número de iterações (épocas) e \bar p é o número médio de atributos diferentes de zero por amostra .

    # Resultados teóricos recentes, no entanto, mostram que o tempo de execução para obter alguma precisão de otimização desejada não aumenta à medida que o tamanho do conjunto de treinamento aumenta. 