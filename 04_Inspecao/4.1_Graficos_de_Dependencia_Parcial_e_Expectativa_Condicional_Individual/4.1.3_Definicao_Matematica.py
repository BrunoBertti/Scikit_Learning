########## 4.1.3. Definição matemática ##########


    # Seja X_S o conjunto de recursos de entrada de interesse (ou seja, o parâmetro de recursos) e seja X_C seu complemento.

    # A dependência parcial da resposta f em um ponto x_S é definida como:

        # \begin{split}pd_{X_S}(x_S) &\overset{def}{=} \mathbb{E}_{X_C}\left[ f(x_S, X_C) \right]\\
        #                &= \int f(x_S, x_C) p(x_C) dx_C,\end{split}


    # onde f(x_S, x_C) é a função de resposta (predict, predict_proba ou decision_function) para uma determinada amostra cujos valores são definidos por x_S para os recursos em X_S e por x_C para os recursos em X_C. Observe que x_S e x_C podem ser tuplas.

    # Calcular esta integral para vários valores de x_S produz um gráfico PDP como acima. Uma linha ICE é definida como um único f(x_{S}, x_{C}^{(i)}) avaliado em x_{S}.


