########## 6.8.6. kernel laplaciano ##########





    # A função laplacian_kernel é uma variante do kernel da função de base radial definida como:

        # k(x, y) = \exp( -\gamma \| x-y \|_1)


    # onde xey são os vetores de entrada e \|x-y\|_1 é a distância de Manhattan entre os vetores de entrada.

    # Ele provou ser útil em ML aplicado a dados sem ruído. Veja, por exemplo Aprendizado de máquina para mecânica quântica em poucas palavras. 