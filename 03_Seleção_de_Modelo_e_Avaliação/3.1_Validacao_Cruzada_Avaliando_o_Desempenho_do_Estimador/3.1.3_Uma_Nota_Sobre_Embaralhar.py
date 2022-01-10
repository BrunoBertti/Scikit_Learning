########## 3.1.3. Uma nota sobre embaralhar ##########

    # Se a ordem dos dados não for arbitrária (por exemplo, as amostras com o mesmo rótulo de classe são contíguas), embaralhá-los primeiro pode ser essencial para obter um resultado de validação cruzada significativo. No entanto, o oposto pode ser verdadeiro se as amostras não forem distribuídas de forma independente e idêntica. Por exemplo, se as amostras correspondem a artigos de notícias e são ordenadas por seu tempo de publicação, então embaralhar os dados provavelmente levará a um modelo que está sobreajuste e uma pontuação de validação inflada: ele será testado em amostras que são artificialmente semelhantes (fechar no tempo) para amostras de treinamento.

    # Alguns iteradores de validação cruzada, como KFold, têm uma opção embutida para embaralhar os índices de dados antes de dividi-los. Observe que: 

        # Isso consome menos memória do que embaralhar os dados diretamente.

        # Por padrão, nenhum embaralhamento ocorre, incluindo para a validação cruzada K (estratificada) realizada especificando cv = some_integer para cross_val_score, grade search, etc. Tenha em mente que train_test_split ainda retorna uma divisão aleatória.

        # O padrão do parâmetro random_state é None, o que significa que o embaralhamento será diferente toda vez que KFold (..., embaralhar = True) for iterado. No entanto, GridSearchCV usará o mesmo embaralhamento para cada conjunto de parâmetros validados por uma única chamada para seu método de ajuste.

        # Para obter resultados idênticos para cada divisão, defina random_state como um inteiro. 

    # Para obter mais detalhes sobre como controlar a aleatoriedade de divisores cv e evitar armadilhas comuns, consulte Controlando a aleatoriedade. 