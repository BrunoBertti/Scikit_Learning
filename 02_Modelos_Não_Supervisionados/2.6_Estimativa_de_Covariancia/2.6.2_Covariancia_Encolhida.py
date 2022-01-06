########## 2.6.2. Covariância Encolhida ##########

    # Apesar de ser um estimador assintoticamente não enviesado da matriz de covariância, o Estimador de Máxima Verossimilhança não é um bom estimador dos autovalores da matriz de covariância, portanto a matriz de precisão obtida de sua inversão não é exata. Às vezes, até ocorre que a matriz de covariância empírica não pode ser invertida por razões numéricas. Para evitar tal problema de inversão, uma transformação da matriz de covariância empírica foi introduzida: o encolhimento.

    # No scikit-learn, essa transformação (com um coeficiente de contração definido pelo usuário) pode ser aplicada diretamente a uma covariância pré-calculada com o método shrunk_covariance. Além disso, um estimador reduzido da covariância pode ser ajustado aos dados com um objeto ShrunkCovariance e seu método ShrunkCovariance.fit. Novamente, os resultados dependem se os dados estão centralizados, portanto, pode-se querer usar o parâmetro assume_centered com precisão.

    # Matematicamente, essa contração consiste em reduzir a razão entre o menor e o maior autovalor da matriz de covariância empírica. Isso pode ser feito simplesmente deslocando cada autovalor de acordo com um determinado deslocamento, que é equivalente a encontrar o Estimador de Máxima Verossimilhança penalizado com l2 da matriz de covariância. Na prática, a redução se reduz a uma simples transformação convexa: \ Sigma _ {\ rmshrunk} = (1- \ alpha) \ hat {\ Sigma} + \ alpha \ frac {{\ rmTr} \ hat {\ Sigma}} { p} \ rm Id


    # Escolhendo a quantidade de redução, \ alpha equivale a definir uma compensação de viés / variância e é discutido abaixo. 


    ## Exemplos:

    ## See Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood for an example on how to fit a ShrunkCovariance object to data. (https://scikit-learn.org/stable/auto_examples/covariance/plot_covariance_estimation.html#sphx-glr-auto-examples-covariance-plot-covariance-estimation-py)




##### 2.6.2.2. Redução de Ledoit-Wolf

    # Em seu artigo 1 de 2004, O. Ledoit e M. Wolf propõem uma fórmula para calcular o coeficiente de contração ótimo \ alpha que minimiza o erro quadrático médio entre a matriz de covariância estimada e real.

    # O estimador Ledoit-Wolf da matriz de covariância pode ser calculado em uma amostra com a função ledoit_wolf do pacote sklearn.covariance, ou pode ser obtido de outra forma ajustando um objeto LedoitWolf à mesma amostra.

    # Nota: Caso quando a matriz de covariância da população é isotrópica
    # É importante notar que quando o número de amostras é muito maior do que o número de características, seria de se esperar que nenhuma redução fosse necessária. A intuição por trás disso é que se a covariância da população for de classificação completa, quando o número da amostra crescer, a covariância da amostra também se tornará definida positiva. Como resultado, nenhum encolhimento seria necessário e o método deveria fazer isso automaticamente.

    # Este, entretanto, não é o caso no procedimento de Ledoit-Wolf quando a covariância da população passa a ser um múltiplo da matriz de identidade. Nesse caso, a estimativa de encolhimento de Ledoit-Wolf se aproxima de 1 conforme o número de amostras aumenta. Isso indica que a estimativa ótima da matriz de covariância no sentido de Ledoit-Wolf é múltipla da identidade. Uma vez que a covariância da população já é um múltiplo da matriz identidade, a solução de Ledoit-Wolf é de fato uma estimativa razoável. 



    ## Exemplos:

    ## See Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood for an example on how to fit a LedoitWolf object to data and for visualizing the performances of the Ledoit-Wolf estimator in terms of likelihood. (https://scikit-learn.org/stable/auto_examples/covariance/plot_covariance_estimation.html#sphx-glr-auto-examples-covariance-plot-covariance-estimation-py)




    ## Referências:

    ## O. Ledoit and M. Wolf, “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices”, Journal of Multivariate Analysis, Volume 88, Issue 2, February 2004, pages 365-411. 


##### 2.6.2.3. Oracle Approximating Shrinkage 



    # Partindo do pressuposto de que os dados têm distribuição gaussiana, Chen et al. 2 derivou uma fórmula que visa escolher um coeficiente de encolhimento que produz um erro quadrático médio menor do que aquele dado pela fórmula de Ledoit e Wolf. O estimador resultante é conhecido como o estimador Oracle Shrinkage Approximating da covariância.

    # O estimador OAS da matriz de covariância pode ser calculado em uma amostra com a função oas do pacote sklearn.covariance, ou pode ser obtido de outra forma ajustando um objeto OAS à mesma amostra. 

        # https://scikit-learn.org/stable/auto_examples/covariance/plot_covariance_estimation.html


    ## Referências:

    ## Chen et al., “Shrinkage Algorithms for MMSE Covariance Estimation”, IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.


    ## Exemplos:

    ## See Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood for an example on how to fit an OAS object to data. (https://scikit-learn.org/stable/auto_examples/covariance/plot_covariance_estimation.html#sphx-glr-auto-examples-covariance-plot-covariance-estimation-py)

    ## See Ledoit-Wolf vs OAS estimation to visualize the Mean Squared Error difference between a LedoitWolf and an OAS estimator of the covariance. (https://scikit-learn.org/stable/auto_examples/covariance/plot_lw_vs_oas.html#sphx-glr-auto-examples-covariance-plot-lw-vs-oas-py)



        # https://scikit-learn.org/stable/auto_examples/covariance/plot_lw_vs_oas.html