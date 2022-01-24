############ 1.8.3. PLS Regressão ############

    # O estimador PLSRegression é semelhante ao PLSCanonical com algoritmo='nipals', com 2 diferenças significativas:

        # na etapa a) no método de potência para calcular u_k e v_k, v_k nunca é normalizado.

        # na etapa c), os alvos Y_k são aproximados usando a projeção de X_k (ou seja, \xi_k) em vez da projeção de Y_k (ou seja, \omega_k). Em outras palavras, o cálculo dos carregamentos é diferente. Como resultado, a deflação na etapa d) também será afetada.

    # Essas duas modificações afetam a saída de previsão e transformação, que não são as mesmas do PLSCanonical. Além disso, enquanto o número de componentes é limitado por min(n_samples, n_features, n_targets) em PLSCanonical, aqui o limite é a classificação de X^TX, ou seja, min(n_samples, n_features).

    # PLSRegression também é conhecido como PLS1 (alvos únicos) e PLS2 (destinos múltiplos). Muito parecido com Lasso, PLSRegression é uma forma de regressão linear regularizada onde o número de componentes controla a força da regularização. 