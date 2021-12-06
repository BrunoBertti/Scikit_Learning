########## 1.1.9 Busca da Correspondecia Perpedicular ##########

    # OrthogonalMatchingPursuit e orthogonal_mp implementa o algoritmo OMP para aproximar o ajuste de um modelo linear com restrições impostas ao número de coeficientes diferentes de zero (isto é, a pseudo-norma 10). 

    # Por ser um método de seleção de recurso avançado, como a regressão de menor ângulo, a busca de correspondência ortogonal pode aproximar o vetor de solução ideal com um número fixo de elementos diferentes de zero: 
        
        # arg min ||y - Xw||22 subject to ||w||w||0 <= n(nonzero)/coeficientes

    # Alternativamente, a busca de correspondência ortogonal pode ter como alvo um erro específico em vez de um número específico de coeficientes diferentes de zero. Isso pode ser expresso como:

        # arg min ||w||0 subject to ||y - Xw||22 <= tol

    # OMP é baseado em um algoritmo guloso que inclui em cada etapa o átomo mais altamente correlacionado com o residual atual. É semelhante ao método mais simples de busca de correspondência (MP), mas melhor em que a cada iteração, o residual é recomputado usando uma projeção ortogonal no espaço dos elementos de dicionário previamente escolhidos. 


    ## Exemplos: https://scikit-learn.org/stable/auto_examples/linear_model/plot_omp.html#sphx-glr-auto-examples-linear-model-plot-omp-py

    ## Referências: https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
    ## http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf S. G. Mallat, Z. Zhang,
    