########## 2.6. Estimativa de covariância ##########


    # Muitos problemas estatísticos requerem a estimativa da matriz de covariância de uma população, que pode ser vista como uma estimativa da forma do gráfico de dispersão do conjunto de dados. Na maioria das vezes, tal estimativa tem que ser feita em uma amostra cujas propriedades (tamanho, estrutura, homogeneidade) têm uma grande influência na qualidade da estimativa. O pacote sklearn.covariance fornece ferramentas para estimar com precisão a matriz de covariância de uma população em várias configurações.

    # Assumimos que as observações são independentes e distribuídas de forma idêntica (i.i.d.). 