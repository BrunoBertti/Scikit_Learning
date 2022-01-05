########## 2.5.5. Análise Fatorial ##########

    # Na aprendizagem não supervisionada, temos apenas um conjunto de dados X = \ {x_1, x_2, \ dots, x_n \}. Como esse conjunto de dados pode ser descrito matematicamente? Um modelo de variável latente contínua muito simples para X é


        # x_i = W h_i + \ mu + \ epsilon


    # O vetor h_i é chamado de “latente” porque não é observado. \ epsilon é considerado um termo de ruído distribuído de acordo com um Gaussiano com média 0 e covariância \ Psi (ou seja, \ epsilon \ sim \ mathcal {N} (0, \ Psi)), \ mu é algum vetor de deslocamento arbitrário. Tal modelo é chamado de “gerador”, pois descreve como x_i é gerado a partir de h_i. Se usarmos todos os x_i's como colunas para formar uma matriz X e todos os h_i's como colunas de uma matriz H, então podemos escrever (com M e E adequadamente definidos):


        # \ mathbf {X} = W \ mathbf {H} + \ mathbf {M} + \ mathbf {E}



    # Em outras palavras, decompomos a matriz X.


    # Se h_i for dado, a equação acima implica automaticamente a seguinte interpretação probabilística:

        # p (x_i | h_i) = \ mathcal {N} (Wh_i + \ mu, \ Psi)


    # Para um modelo probabilístico completo, também precisamos de uma distribuição anterior para a variável latente h. A suposição mais direta (com base nas boas propriedades da distribuição gaussiana) é h \ sim \ mathcal {N} (0, \ mathbf {I}). Isso resulta em um Gaussiano como a distribuição marginal de x:


        # p (x) = \ mathcal {N} (\ mu, WW ^ T + \ Psi)



    # Agora, sem quaisquer suposições adicionais, a ideia de ter uma variável latente h seria supérflua –x pode ser completamente modelada com uma média e uma covariância. Precisamos impor uma estrutura mais específica a um desses dois parâmetros. Uma suposição adicional simples diz respeito à estrutura da covariância do erro \ Psi:



        # \ Psi = \ sigma ^ 2 \ mathbf {I}: esta suposição leva ao modelo probabilístico de PCA.

        # \ Psi = \ mathrm {diag} (\ psi_1, \ psi_2, \ dots, \ psi_n): Este modelo é denominado FactorAnalysis, um modelo estatístico clássico. A matriz W é às vezes chamada de “matriz de carga fatorial”.


    # Ambos os modelos estimam essencialmente uma Gaussiana com uma matriz de covariância de baixa classificação. Como ambos os modelos são probabilísticos, eles podem ser integrados em modelos mais complexos, por ex. Mistura de analisadores de fator. Obtém-se modelos muito diferentes (por exemplo, FastICA) se as anteriores não gaussianas nas variáveis ​​latentes forem assumidas.

    # A análise fatorial pode produzir componentes semelhantes (as colunas de sua matriz de carregamento) ao PCA. No entanto, não se pode fazer nenhuma declaração geral sobre esses componentes (por exemplo, se eles são ortogonais): 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html

    # A principal vantagem da Análise Fatorial sobre o PCA é que ela pode modelar a variação em todas as direções do espaço de entrada de forma independente (ruído heterocedástico): 


        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html

    
    # Isso permite uma melhor seleção de modelo do que PCA probabilístico na presença de ruído heterocedástico: 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html


    # A Análise Fatorial costuma ser seguida por uma rotação dos fatores (com a rotação do parâmetro), geralmente para melhorar a interpretabilidade. Por exemplo, a rotação Varimax maximiza a soma das variâncias dos carregamentos quadrados, ou seja, tende a produzir fatores mais esparsos, que são influenciados por apenas algumas características cada (a "estrutura simples"). Veja, por exemplo, o primeiro exemplo abaixo. 




    ## Exemplos:
 
    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_varimax_fa.html#sphx-glr-auto-examples-decomposition-plot-varimax-fa-py

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py