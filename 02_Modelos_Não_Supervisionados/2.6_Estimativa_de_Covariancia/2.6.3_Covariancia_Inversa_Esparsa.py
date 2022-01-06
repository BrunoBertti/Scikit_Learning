########## 2.6.3. Covariância inversa esparsa ##########


    # A matriz inversa da matriz de covariância, freqüentemente chamada de matriz de precisão, é proporcional à matriz de correlação parcial. Ele dá a relação de independência parcial. Em outras palavras, se dois recursos são independentes condicionalmente uns dos outros, o coeficiente correspondente na matriz de precisão será zero. É por isso que faz sentido estimar uma matriz de precisão esparsa: a estimativa da matriz de covariância é mais bem condicionada por aprender relações de independência a partir dos dados. Isso é conhecido como seleção de covariância.

    # Na situação de pequenas amostras, em que n_amostras é da ordem de n_características ou menores, os estimadores de covariância inversa esparsos tendem a funcionar melhor do que os estimadores de covariância reduzida. Porém, na situação oposta, ou para dados muito correlacionados, eles podem ser numericamente instáveis. Além disso, ao contrário dos estimadores de encolhimento, os estimadores esparsos são capazes de recuperar estruturas fora da diagonal.

    # O estimador GraphicalLasso usa uma penalidade l1 para impor a esparsidade na matriz de precisão: quanto mais alto seu parâmetro alfa, mais esparsa é a matriz de precisão. O objeto GraphicalLassoCV correspondente usa validação cruzada para definir automaticamente o parâmetro alfa. 

        # https://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html

    
    # Nota: recuperação de estrutura
    # Recuperar uma estrutura gráfica de correlações nos dados é uma coisa desafiadora. Se você estiver interessado em tal recuperação, tenha em mente que:

        # A recuperação é mais fácil de uma matriz de correlação do que uma matriz de covariância: padronize suas observações antes de executar GraphicalLasso

        # Se o grafo subjacente tiver nós com muito mais conexões do que o nó médio, o algoritmo perderá algumas dessas conexões.

        # Se o seu número de observações não for grande em comparação com o número de arestas do gráfico subjacente, você não o recuperará.

        # Mesmo se você estiver em condições de recuperação favoráveis, o parâmetro alfa escolhido por validação cruzada (por exemplo, usando o objeto GraphicalLassoCV) levará à seleção de muitas bordas. No entanto, as arestas relevantes terão pesos mais pesados do que as irrelevantes. 


    

    # A formulação matemática é a seguinte: 

        # \hat{K} = \mathrm{argmin}_K \big(
        #    \mathrm{tr} S K - \mathrm{log} \mathrm{det} K
        #    + \alpha \|K\|_1
        #    \big)


    # Onde K é a matriz de precisão a ser estimada e S é a matriz de covariância da amostra. \ | K \ | _1 é a soma dos valores absolutos dos coeficientes fora da diagonal de K. O algoritmo empregado para resolver este problema é o algoritmo GLasso, do artigo de Friedman 2008 Biostatistics. É o mesmo algoritmo do pacote R glasso. 



    ## Exemplos:

    ## Sparse inverse covariance estimation: example on synthetic data showing some recovery of a structure, and comparing to other covariance estimators. (https://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html#sphx-glr-auto-examples-covariance-plot-sparse-cov-py)

    ## Visualizing the stock market structure: example on real stock market data, finding which symbols are most linked. (https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#sphx-glr-auto-examples-applications-plot-stock-market-py)



    ## Referências:

    ## Friedman et al, “Sparse inverse covariance estimation with the graphical lasso”, Biostatistics 9, pp 432, 2008 (https://biostatistics.oxfordjournals.org/content/9/3/432.short)
