############ 1.8.2. PLSSVD ############

    # PLSSVD é uma versão simplificada do PLSCanonical descrito anteriormente: em vez de deflacionar iterativamente as matrizes X_k e Y_k, PLSSVD calcula o SVD de C = X^TY apenas uma vez e armazena os n_components vetores singulares correspondentes aos maiores valores singulares nas matrizes U e V, correspondente aos atributos x_weights_ e y_weights_. Aqui, os dados transformados são simplesmente transformados(X) = XU e transformados(Y) = YV.

    # Se n_components == 1, PLSSVD e PLSCanonical são estritamente equivalentes. 