########## 6.7.6. Detalhes matemáticos  ##########




    # Métodos de kernel como máquinas de vetor de suporte ou PCA kernelizado dependem de uma propriedade de reproduzir espaços de Hilbert do kernel. Para qualquer função de kernel definida positiva k (um chamado kernel de Mercer), é garantido que existe um mapeamento \phi em um espaço de Hilbert \mathcal{H}, tal que 


        # k(x,y) = \langle \phi(x), \phi(y) \rangle




    # Onde \langle \cdot, \cdot \rangle denota o produto interno no espaço de Hilbert.


    # Se um algoritmo, como uma máquina de vetor de suporte linear ou PCA, depende apenas do produto escalar dos pontos de dados x_i, pode-se usar o valor de k(x_i, x_j), que corresponde à aplicação do algoritmo aos pontos de dados mapeados \ phi(x_i). A vantagem de usar k é que o mapeamento \phi nunca precisa ser calculado explicitamente, permitindo grandes recursos arbitrários (mesmo infinitos).


    # Uma desvantagem dos métodos do kernel é que pode ser necessário armazenar muitos valores do kernel k(x_i, x_j) durante a otimização. Se um classificador kernelizado for aplicado a novos dados y_j, k(x_i, y_j) precisa ser calculado para fazer previsões, possivelmente para muitos x_i diferentes no conjunto de treinamento.

    # As classes deste submódulo permitem aproximar o \phi de embutimento, trabalhando assim explicitamente com as representações \phi(x_i), o que dispensa a necessidade de aplicar o kernel ou armazenar exemplos de treinamento. 





    ## Referências:


    ## RR2007(1,2) “Random features for large-scale kernel machines” Rahimi, A. and Recht, B. - Advances in neural information processing 2007, (https://www.robots.ox.ac.uk/~vgg/rg/papers/randomfeatures.pdf)

    ## LS2010 “Random Fourier approximations for skewed multiplicative histogram kernels” Random Fourier approximations for skewed multiplicative histogram kernels - Lecture Notes for Computer Sciencd (DAGM) (http://www.maths.lth.se/matematiklth/personal/sminchis/papers/lis_dagm10.pdf)

    ## VZ2010(1,2) “Efficient additive kernels via explicit feature maps” Vedaldi, A. and Zisserman, A. - Computer Vision and Pattern Recognition 2010 (https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)

    ## VVZ2010 “Generalized RBF feature maps for Efficient Detection” Vempati, S. and Vedaldi, A. and Zisserman, A. and Jawahar, CV - 2010 (https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf)

    ## PP2013 “Fast and scalable polynomial kernels via explicit feature maps” Pham, N., & Pagh, R. - 2013 (https://doi.org/10.1145/2487575.2487591)

    ## CCF2002 “Finding frequent items in data streams” Charikar, M., Chen, K., & Farach-Colton - 2002 (http://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarCF.pdf)

    ## WIKICS “Wikipedia: Count sketch” (https://en.wikipedia.org/wiki/Count_sketch)


   
   
   