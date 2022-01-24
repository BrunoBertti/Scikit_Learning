############ 1.8.4. Análise de correlação canônica  ############

    # A Análise de Correlação Canônica foi desenvolvida antes e independentemente do PLS. Mas acontece que o CCA é um caso especial de PLS, e corresponde ao PLS em “Modo B” na literatura.

    # O CCA difere do PLSCanonical na maneira como os pesos u_k e v_k são calculados no método de potência da etapa a). Os detalhes podem ser encontrados na seção 10 de 1.

    # Como o CCA envolve a inversão de X_k^TX_k e Y_k^TY_k, esse estimador pode ser instável se o número de feições ou alvos for maior que o número de amostras. 



    ## Referências:
    
    ## 1(1,2,3,4) A survey of Partial Least Squares (PLS) methods, with emphasis on the two-block case JA Wegelin (https://www.stat.washington.edu/research/reports/2000/tr371.pdf)




    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_compare_cross_decomposition.html#sphx-glr-auto-examples-cross-decomposition-plot-compare-cross-decomposition-py

    ## https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py