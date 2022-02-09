########## 6.8.3. Kernel polinomial  ##########



    # A função polynomial_kernel calcula o kernel polinomial de grau d entre dois vetores. O kernel polinomial representa a semelhança entre dois vetores. Conceitualmente, os núcleos polinomiais consideram não apenas a similaridade entre vetores sob a mesma dimensão, mas também entre dimensões. Quando usado em algoritmos de aprendizado de máquina, isso permite considerar a interação de recursos.

    # O kernel polinomial é definido como: 

        # k(x, y) = (\gamma x^\top y +c_0)^d

    # Onde:

        # x, y são os vetores de entrada

        # d é o grau do kernel

    # Se c_0 = 0 o kernel é dito homogêneo. 