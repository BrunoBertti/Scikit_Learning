########## 6.8.1. Semelhança de cosseno ##########


    # cosine_similarity calcula o produto escalar normalizado por L2 de vetores. Ou seja, se x e y são vetores linha, sua similaridade de cosseno k é definida como:


        # k(x, y) = \frac{x y^\top}{\|x\| \|y\|}


    # Isso é chamado de similaridade de cosseno, porque a normalização euclidiana (L2) projeta os vetores na esfera unitária, e seu produto escalar é então o cosseno do ângulo entre os pontos denotados pelos vetores.

    # Este kernel é uma escolha popular para calcular a similaridade de documentos representados como vetores tf-idf. cosine_similarity aceita matrizes scipy.sparse. (Observe que a funcionalidade tf-idf em sklearn.feature_extraction.text pode produzir vetores normalizados, caso em que cosine_similarity é equivalente a linear_kernel, apenas mais lento.) 



    ## Referências:

    ## C.D. Manning, P. Raghavan and H. Schütze (2008). Introduction to Information Retrieval. Cambridge University Press. (https://nlp.stanford.edu/IR-book/html/htmledition/the-vector-space-model-for-scoring-1.html)