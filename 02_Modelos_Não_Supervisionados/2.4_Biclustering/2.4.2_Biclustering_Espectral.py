########## 2.4.2. Biclustering espectral ##########

    # O algoritmo SpectralBiclustering assume que a matriz de dados de entrada tem uma estrutura quadriculada oculta. As linhas e colunas de uma matriz com esta estrutura podem ser particionadas de forma que as entradas de qualquer bicluster no produto cartesiano de clusters de linha e clusters de coluna sejam aproximadamente constantes. Por exemplo, se houver duas partições de linha e três partições de coluna, cada linha pertencerá a três biclusters e cada coluna pertencerá a dois biclusters.

    # O algoritmo particiona as linhas e colunas de uma matriz de modo que uma matriz quadrada de constante bloco correspondente forneça uma boa aproximação da matriz original. 


##### 2.4.2.1. Formulação matemática 

    # A matriz de entrada A é primeiro normalizada para tornar o padrão xadrez mais óbvio. Existem três métodos possíveis: 

    # Normalização independente de linha e coluna, como no Spectral Co-Clustering. Este método faz com que a soma das linhas seja uma constante e a soma das colunas seja uma constante diferente.

    # Bistochastização: normalização repetida de linhas e colunas até a convergência. Este método faz com que as linhas e colunas sejam somadas à mesma constante.

    # Normalização do log: o log da matriz de dados é calculado: L =\ log A. Então a média da coluna \ overline {L_ {i \ cdot}}, média da linha \ overline {L _ {\ cdot j}}, e média geral \ overline {L _ {\ cdot\ cdot}} de L são calculados. A matriz final é calculada de acordo com a fórmula


        # K_ {ij} = L_ {ij} - \ overline {L_ {i \ cdot}} - \ overline {L _ {\ cdot
        # j}} + \ overline {L _ {\ cdot \ cdot}}

    # Após a normalização, os primeiros vetores singulares são calculados, assim como no algoritmo Spectral Co-Clustering.


    # Se a normalização de log foi usada, todos os vetores singulares são significativos. No entanto, se normalização independente ou bistochastização foram usados, os primeiros vetores singulares, u_1 e v_1. são descartados. A partir de agora, os “primeiros” vetores singulares referem-se a u_2 \ pontos u_ {p + 1} e v_2 \ pontos v_ {p + 1} exceto no caso de normalização logarítmica.


    # Dados esses vetores singulares, eles são classificados de acordo com os quais podem ser mais bem aproximados por um vetor constante por partes. As aproximações para cada vetor são encontradas usando k-médias unidimensionais e pontuadas usando a distância euclidiana. Alguns subconjuntos do melhor vetor singular esquerdo e direito são selecionados. Em seguida, os dados são projetados para este melhor subconjunto de vetores singulares e agrupados.


    # Por exemplo, se p vetores singulares foram calculados, os melhores q são encontrados conforme descrito, onde q <p. Seja U a matriz com colunas os q melhores vetores singulares à esquerda e, da mesma forma, V à direita. Para particionar as linhas, as linhas de A são projetadas em um espaço dimensional q: A * V. Tratar as m linhas dessa matriz m \ vezes q como amostras e agrupar usando k-médias produz os rótulos das linhas. Da mesma forma, projetar as colunas em A ^ {\ top} * U e agrupar esta matriz n \ vezes q produz os rótulos das colunas. 


    ## Exemplos:

    ## A demo of the Spectral Biclustering algorithm: a simple example showing how to generate a checkerboard matrix and bicluster it. (https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html#sphx-glr-auto-examples-bicluster-plot-spectral-biclustering-py)




    ## Referências:

    ## Kluger, Yuval, et. al., 2003. Spectral biclustering of microarray data: coclustering genes and conditions. (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.135.1608)