########## 6.7. Aproximação do kernel ##########




    # Este submódulo contém funções que aproximam os mapeamentos de recursos que correspondem a determinados kernels, pois são usados, por exemplo, em máquinas de vetor de suporte (consulte Máquinas de vetor de suporte). As funções de recurso a seguir realizam transformações não lineares da entrada, que podem servir como base para classificação linear ou outros algoritmos.

    # A vantagem de usar mapas de recursos explícitos aproximados em comparação com o truque do kernel, que faz uso de mapas de recursos implicitamente, é que os mapeamentos explícitos podem ser mais adequados para aprendizado online e podem reduzir significativamente o custo de aprendizado com conjuntos de dados muito grandes. SVMs kernelizadas padrão não se adaptam bem a grandes conjuntos de dados, mas usando um mapa de kernel aproximado é possível usar SVMs lineares muito mais eficientes. Em particular, a combinação de aproximações do mapa do kernel com o SGDClassifier pode possibilitar o aprendizado não linear em grandes conjuntos de dados.

    # Como não houve muito trabalho empírico usando embeddings aproximados, é aconselhável comparar os resultados com métodos de kernel exatos quando possível.

    # Veja também: Regressão polinomial: estendendo modelos lineares com funções de base para uma transformação polinomial exata. 