########## 2.5.2. Análise de componentes principais do kernel (kPCA) ##########

##### 2.5.2.1. PCA Exato do Kernel

    # KernelPCA é uma extensão do PCA que alcança a redução da dimensionalidade não linear por meio do uso de kernels (consulte métricas Pairwise, Affinities and Kernels) [Scholkopf1997]. Ele tem muitas aplicações, incluindo redução de ruído, compressão e predição estruturada (estimativa de dependência de kernel). KernelPCA oferece suporte a transform e inverse_transform. 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html


    # Nota: KernelPCA.inverse_transform depende de um núcleo do kernel para aprender as amostras de mapeamento de função da base do PCA para o espaço de recurso original [Bakir2004]. Assim, a reconstrução obtida com KernelPCA.inverse_transform é uma aproximação. Veja o exemplo no link abaixo para mais detalhes. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py



    ## Referências:

    ## Scholkopf1997 Schölkopf, Bernhard, Alexander Smola, and Klaus-Robert Müller. “Kernel principal component analysis.” International conference on artificial neural networks. Springer, Berlin, Heidelberg, 1997. (https://people.eecs.berkeley.edu/~wainwrig/stat241b/scholkopf_kernel.pdf)

    ## Bakir2004 Bakır, Gökhan H., Jason Weston, and Bernhard Schölkopf. “Learning to find pre-images.” Advances in neural information processing systems 16 (2004): 449-456. (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5164&rep=rep1&type=pdf)


##### 2.5.2.2. Escolha de solucionador para Kernel PCA 

    # Enquanto no PCA o número de componentes é limitado pelo número de recursos, no KernelPCA o número de componentes é limitado pelo número de amostras. Muitos conjuntos de dados do mundo real têm um grande número de amostras! Nestes casos, encontrar todos os componentes com um kPCA completo é uma perda de tempo de computação, uma vez que os dados são descritos principalmente pelos primeiros componentes (por exemplo, n_components <= 100). Em outras palavras, a matriz de Gram centrada que é decomposta no processo de ajuste PCA do Kernel tem uma classificação efetiva muito menor do que seu tamanho. Esta é uma situação em que os eigensolvers aproximados podem fornecer aceleração com perda de precisão muito baixa.

    # O parâmetro opcional eigen_solver = 'randomized' pode ser usado para reduzir significativamente o tempo de cálculo quando o número de n_components solicitados é pequeno em comparação com o número de amostras. Ele se baseia em métodos de decomposição aleatória para encontrar uma solução aproximada em um tempo mais curto.


    # A complexidade de tempo do KernelPCA aleatório é O (n _ {\ mathrm {amostras}} ^ 2 \ cdot n _ {\ mathrm {componentes}}) em vez de O (n _ {\ mathrm {amostras}} ^ 3) para o método exato implementado com eigen_solver = 'denso'.


    # A pegada de memória do KernelPCA randomizado também é proporcional a 2 \ cdot n _ {\ mathrm {samples}} \ cdot n _ {\ mathrm {componentes}} em vez de n _ {\ mathrm {samples}} ^ 2 para o método exato.


    # Nota: esta técnica é a mesma que no PCA usando SVD aleatório.

    # Além dos dois solucionadores acima, eigen_solver = 'arpack' pode ser usado como uma maneira alternativa de obter uma decomposição aproximada. Na prática, esse método só fornece tempos de execução razoáveis ​​quando o número de componentes a localizar é extremamente pequeno. É habilitado por padrão quando o número desejado de componentes é menor que 10 (estrito) e o número de amostras é maior que 200 (estrito). Consulte KernelPCA para obter detalhes. 



    ## Referências:

    ## dense solver: scipy.linalg.eigh documentation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html)

    ## randomized solver:

        ### Algorithm 4.3 in “Finding structure with randomness: Stochastic algorithms for constructing approximate matrix decompositions” Halko, et al., 2009 (https://arxiv.org/abs/0909.4061)

        ### “An implementation of a randomized algorithm for principal component analysis” A. Szlam et al. 2014 (https://arxiv.org/abs/0909.4061)

    ## arpack solver: scipy.sparse.linalg.eigsh documentation R. B. Lehoucq, D. C. Sorensen, and C. Yang, 1998 (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)