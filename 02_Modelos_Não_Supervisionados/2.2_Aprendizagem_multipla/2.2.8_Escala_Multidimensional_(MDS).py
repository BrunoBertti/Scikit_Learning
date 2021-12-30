########## 2.2.8. Escala multidimensional (MDS) ##########

    # O escalonamento multidimensional (MDS) busca uma representação de baixa dimensão dos dados em que as distâncias respeitem bem as distâncias no espaço de alta dimensão original.

    # Em geral, MDS é uma técnica usada para analisar dados de similaridade ou dissimilaridade. Ele tenta modelar dados de similaridade ou dissimilaridade como distâncias em espaços geométricos. Os dados podem ser classificações de similaridade entre objetos, frequências de interação de moléculas ou índices comerciais entre países.

    # Existem dois tipos de algoritmo MDS: métrico e não métrico. No scikit-learn, a classe MDS implementa ambos. No Metric MDS, a matriz de similaridade de entrada surge de uma métrica (e, portanto, respeita a desigualdade triangular), as distâncias entre os dois pontos de saída são então definidas para ser o mais próximo possível dos dados de similaridade ou dissimilaridade. Na versão não métrica, os algoritmos tentarão preservar a ordem das distâncias e, assim, buscar uma relação monotônica entre as distâncias no espaço embutido e as semelhanças / dissimilaridades. 

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

    # Seja S a matriz de similaridade e X as coordenadas dos n pontos de entrada. Disparidades \ hat {d} _ {ij} são transformações das semelhanças escolhidas de algumas maneiras ótimas. O objetivo, chamado de estresse, é então definido por \ sum_ {i <j} d_ {ij} (X) - \ hat {d} _ {ij} (X) 



##### 2.2.8.1. MDS métrico

    # O modelo métrico mais simples de MDS, chamado MDS absoluto, as disparidades são definidas por \ hat {d} _ {ij} = S_ {ij}. Com MDS absoluto, o valor S_ {ij} deve então corresponder exatamente à distância entre o ponto e no ponto de incorporação.

    # Mais comumente, as disparidades são definidas como \ hat {d} _ {ij} = b S_ {ij}

##### 2.2.8.2. MDS não métrico 

    # O MDS não métrico concentra-se na ordenação dos dados. Se S_ {ij} <S_ {jk}, então a incorporação deve impor d_ {ij} <d_ {jk}. Um algoritmo simples para impor isso é usar uma regressão monotônica de d_ {ij} em S_ {ij}, produzindo disparidades \ hat {d} _ {ij} na mesma ordem que S_ {ij}.

    # Uma solução trivial para esse problema é definir todos os pontos na origem. Para evitar isso, as disparidades \ hat {d} _ {ij} são normalizadas.

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html

    

    ## Referências:

    ## “Modern Multidimensional Scaling - Theory and Applications” Borg, I.; Groenen P. Springer Series in Statistics (1997) (https://www.springer.com/fr/book/9780387251509)

    ## “Nonmetric multidimensional scaling: a numerical method” Kruskal, J. Psychometrika, 29 (1964) (https://link.springer.com/article/10.1007%2FBF02289694)

    ## “Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis” Kruskal, J. Psychometrika, 29, (1964) (https://link.springer.com/article/10.1007%2FBF02289565)