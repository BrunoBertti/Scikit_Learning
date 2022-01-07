########## 2.7.4. Detecção de novidade com fator de outlier local  ##########


    # Para usar neighbours.LocalOutlierFactor para detecção de novidade, ou seja, prever rótulos ou calcular a pontuação de anormalidade de novos dados não vistos, você precisa instanciar o estimador com o parâmetro de novidade definido como True antes de ajustar o estimador: 

        # lof = LocalOutlierFactor(novelty=True)
        # lof.fit(X_train)

    # Observe que fit_predict não está disponível neste caso. 

    # Aviso: detecção de novidade com fator de outlier local
    # Quando a novidade é definida como True, esteja ciente de que você só deve usar predizer, decision_function e score_samples em novos dados não vistos e não nas amostras de treinamento, pois isso levaria a resultados errados. As pontuações de anormalidade das amostras de treinamento estão sempre acessíveis por meio do atributo negative_outlier_factor_. 


    # A detecção de novidades com Fator de Outlier Local é ilustrada abaixo. 

        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html