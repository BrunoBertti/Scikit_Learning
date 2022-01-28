########## 4.2.2. Relação com a importância baseada em impurezas em árvores ##########

    # Modelos baseados em árvore fornecem uma medida alternativa de importância de recursos com base na diminuição média em impureza (MDI). A impureza é quantificada pelo critério de divisão das árvores de decisão (Gini, Entropia ou Erro Quadrado Médio). No entanto, esse método pode dar alta importância a recursos que podem não ser preditivos em dados não vistos quando o modelo está sobreajustado. A importância do recurso baseado em permutação, por outro lado, evita esse problema, pois pode ser calculado em dados não vistos.

    # Além disso, a importância de características baseadas em impureza para árvores são fortemente tendenciosas e favorecem características de alta cardinalidade (normalmente características numéricas) sobre características de baixa cardinalidade, como características binárias ou variáveis ​​categóricas com um pequeno número de categorias possíveis.

    # As importâncias dos recursos baseados em permutação não exibem esse viés. Além disso, a importância do recurso de permutação pode ser calculada como métrica de desempenho nas previsões de previsões do modelo e pode ser usada para analisar qualquer classe de modelo (não apenas modelos baseados em árvore).

    # O exemplo a seguir destaca as limitações da importância do recurso baseado em impureza em contraste com a importância do recurso baseado em permutação: Importância de Permutação vs Importância de Recurso de Floresta Aleatória (MDI). 