########## 6.8.4. Kernel sigmoide ##########



    # A função sigmoid_kernel calcula o kernel sigmoid entre dois vetores. O kernel sigmoid também é conhecido como tangente hiperbólica, ou Multilayer Perceptron (porque, no campo da rede neural, é frequentemente usado como função de ativação de neurônios). É definido como: 

        # k(x, y) = \tanh( \gamma x^\top y + c_0)

    # Onde:

        # x, y são os vetores de entrada

        # \gamma é conhecido como inclinação

        # c_0 é conhecido como interceptar 

