########## 6.6.3. Projeção aleatória esparsa  ##########



    # A SparseRandomProjection reduz a dimensionalidade projetando o espaço de entrada original usando uma matriz aleatória esparsa.

    # Matrizes aleatórias esparsas são uma alternativa à matriz de projeção aleatória gaussiana densa que garante qualidade de incorporação semelhante, sendo muito mais eficiente em memória e permitindo computação mais rápida dos dados projetados.

    # Se definirmos s = 1 / densidade, os elementos da matriz aleatória são extraídos de 


        # \begin{split}\left\{
        # \begin{array}{c c l}
        # -\sqrt{\frac{s}{n_{\text{components}}}} & & 1 / 2s\\
        # 0 &\text{with probability}  & 1 - 1 / s \\
        # +\sqrt{\frac{s}{n_{\text{components}}}} & & 1 / 2s\\
        # \end{array}
        # \right.\end{split}



    # onde n_{\text{components}} é o tamanho do subespaço projetado. Por padrão, a densidade de elementos diferentes de zero é definida como a densidade mínima recomendada por Ping Li et al.: 1 / \sqrt{n_{\text{features}}}.

    # Aqui um pequeno trecho que ilustra como usar o transformador de projeção aleatória esparsa: 



import numpy as np
from sklearn import random_projection
X = np.random.rand(100, 10000)
transformer = random_projection.SparseRandomProjection()
X_new = transformer.fit_transform(X)
X_new.shape



    ## Referências:

    ## D. Achlioptas. 2003. Database-friendly random projections: Johnson-Lindenstrauss with binary coins. Journal of Computer and System Sciences 66 (2003) 671–687 (http://www.cs.ucsc.edu/~optas/papers/jl.pdf)

    ## Ping Li, Trevor J. Hastie, and Kenneth W. Church. 2006. Very sparse random projections. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD ‘06). ACM, New York, NY, USA, 287-296. (https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf)