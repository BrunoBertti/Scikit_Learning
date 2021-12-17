########## 1.10.6. Algoritmos de árvore: ID3, C4.5, C5.0 e CART  ##########

    # Quais são os vários algoritmos de árvore de decisão e como eles diferem uns dos outros? Qual é implementado no scikit-learn?

    # ID3 (Iterative Dichotomiser 3) foi desenvolvido em 1986 por Ross Quinlan. O algoritmo cria uma árvore de múltiplas vias, encontrando para cada nó (ou seja, de forma gananciosa) a característica categórica que produzirá o maior ganho de informação para alvos categóricos. As árvores crescem até seu tamanho máximo e, em seguida, uma etapa de poda é geralmente aplicada para melhorar a capacidade da árvore de generalizar para dados invisíveis.

    # C4.5 é o sucessor de ID3 e removeu a restrição de que os recursos devem ser categóricos, definindo dinamicamente um atributo discreto (com base em variáveis ​​numéricas) que particiona o valor do atributo contínuo em um conjunto discreto de intervalos. C4.5 converte as árvores treinadas (ou seja, a saída do algoritmo ID3) em conjuntos de regras se-então. A precisão de cada regra é então avaliada para determinar a ordem em que devem ser aplicadas. A poda é feita removendo a pré-condição de uma regra se a precisão da regra melhorar sem ela.

    # C5.0 é a versão mais recente da Quinlan lançada sob uma licença proprietária. Ele usa menos memória e cria conjuntos de regras menores do que o C4.5, sendo mais preciso.

    # CART (árvores de classificação e regressão) é muito semelhante a C4.5, mas difere porque oferece suporte a variáveis ​​de destino numéricas (regressão) e não calcula conjuntos de regras. O CART constrói árvores binárias usando o recurso e o limite que geram o maior ganho de informação em cada nó.

    # scikit-learn usa uma versão otimizada do algoritmo CART; no entanto, a implementação do scikit-learn não oferece suporte a variáveis ​​categóricas por enquanto. 