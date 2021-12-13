########## 1.4.4 Complexidades ##########

    # As máquinas de vetores de suporte são ferramentas poderosas, mas seus requisitos de computação e armazenamento aumentam rapidamente com o número de vetores de treinamento. O núcleo de um SVM é um problema de programação quadrática (QP), separando vetores de suporte do resto dos dados de treinamento. O solucionador de QP usado pela implementação baseada em libsvm escala entre O (n_ {variáveis} \ vezes n_ {amostras} ^ 2) e variáveis} \ vezes n_ {amostras} ^ 3) dependendo da eficiência do cache libsvm é usado na prática (dependente do conjunto de dados). Se os dados forem muito esparsos, n_ {variáveis} deve ser substituído pelo número médio de variáveis diferentes de zero em um vetor de amostra.

    # Para o caso linear, o algoritmo usado em LinearSVC pela implementação liblinear é muito mais eficiente do que sua contraparte SVC baseada em libsvm e pode escalar quase linearmente para milhões de amostras e / ou variáveis.  