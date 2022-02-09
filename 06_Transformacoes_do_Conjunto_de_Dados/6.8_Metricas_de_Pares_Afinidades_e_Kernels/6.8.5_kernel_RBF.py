########## 6.8.5. kernel RBF ##########



    # A função rbf_kernel calcula o kernel da função de base radial (RBF) entre dois vetores. Este kernel é definido como: 

        # k(x, y) = \exp( -\gamma \| x-y \|^2)

    # onde x e y são os vetores de entrada. Se \gamma = \sigma^{-2} o kernel é conhecido como kernel gaussiano de variância \sigma^2. 

