########## 6.8.2. Kernel linear ##########


    # A função linear_kernel calcula o kernel linear, ou seja, um caso especial de polynomial_kernel com grau=1 e coef0=0 (homogêneo). Se x e y são vetores de coluna, seu kernel linear é: 

        # k(x, y) = x^\top y