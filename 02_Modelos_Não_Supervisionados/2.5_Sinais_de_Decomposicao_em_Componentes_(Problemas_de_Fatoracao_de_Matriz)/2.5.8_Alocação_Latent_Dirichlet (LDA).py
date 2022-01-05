########## 2.5.8. Alocação Latent Dirichlet (LDA)  ##########

    # Latent Dirichlet Allocation é um modelo probabilístico generativo para coleções de conjuntos de dados discretos, como corpora de texto. É também um modelo de tópico usado para descobrir tópicos abstratos de uma coleção de documentos.

    # O modelo gráfico de LDA é um modelo gerador de três níveis: 

        # https://scikit-learn.org/stable/_images/lda_model_graph.png

    # Nota sobre as notações apresentadas no modelo gráfico acima, que podem ser encontradas em Hoffman et al. (2013): 

        # O corpus é uma coleção de documentos D.

        # Um documento é uma sequência de N palavras.

        # Existem K tópicos no corpus.

        # As caixas representam amostragem repetida. 

    # No modelo gráfico, cada nó é uma variável aleatória e tem uma função no processo gerador. Um nó sombreado indica uma variável observada e um nó não sombreado indica uma variável oculta (latente). Nesse caso, as palavras do corpus são os únicos dados que observamos. As variáveis latentes determinam a mistura aleatória de tópicos no corpus e a distribuição das palavras nos documentos. O objetivo do LDA é usar as palavras observadas para inferir a estrutura do tópico oculto. 


    # Ao modelar corpora de texto, o modelo assume o seguinte processo gerador para um corpus com documentos D e K tópicos, com K correspondendo a n_components na API: 

        # 1- Para cada tópico k \ in K, desenhe \ beta_k \ sim\ mathrm {Dirichlet} (\ eta). Isso fornece uma distribuição sobre as palavras, ou seja, a probabilidade de uma palavra aparecer no tópico k. \ eta corresponde a topic_word_prior.

        # 2- Para cada documento d \ in D, desenhe as proporções do tópico \ theta_d \ sim \ mathrm {Dirichlet} (\ alpha). \ alpha corresponde a doc_topic_prior.

        # 3- Para cada palavra i no documento d: 


            # 1- Desenhe a atribuição de tópico z_ {di} \ sim \ mathrm {Multinomial}(\ theta_d)

            # 2- Desenhe a palavra observada w_ {ij} \ sim \ mathrm {Multinomial}(\ beta_ {z_ {di}}) 


    # Para estimativa de parâmetro, a distribuição posterior é: 

        # p(z, \theta, \beta |w, \alpha, \eta) =  \frac{p(z, \theta, \beta|\alpha, \eta)}{p(w|\alpha, \eta)}

    # Uma vez que a posterior é intratável, o método variacional Bayesiano usa uma distribuição mais simples q (z, \ theta, \ beta | \ lambda, \ phi, \ gamma) para se aproximar dela, e esses parâmetros variacionais \ lambda, \ phi, \ gamma são otimizados para maximizar o Limite inferior de evidência (ELBO):

        # \ log \: P (w | \ alpha, \ eta) \ geq L (w, \ phi, \ gamma, \ lambda) \ overset {\ triangle} {=}   E_ {q} [\ log \: p (w, z, \ theta, \ beta | \ alpha, \ eta)] - E_ {q} [\ log \: q (z, \ theta, \ beta)] 

    # Maximizar o ELBO é equivalente a minimizar a divergência de Kullback-Leibler (KL) entre q (z, \ theta, \ beta) e o verdadeiro p posterior (z, \ theta, \ beta | w, \ alpha, \ eta).

    # LatentDirichletAllocation implementa o algoritmo variacional Bayes online e suporta métodos de atualização online e em lote. Enquanto o método de lote atualiza as variáveis ​​variacionais após cada passagem completa pelos dados, o método online atualiza as variáveis ​​variacionais dos pontos de dados do minilote.

    # Observação: embora o método online tenha garantia de convergência para um ponto ótimo local, a qualidade do ponto ótimo e a velocidade de convergência podem depender do tamanho do minilote e dos atributos relacionados à configuração da taxa de aprendizagem.

    # Quando LatentDirichletAllocation é aplicado em uma matriz “documento-termo”, a matriz será decomposta em uma matriz “tópico-termo” e uma matriz “documento-tópico”. Enquanto a matriz “tópico-termo” é armazenada como componentes_ no modelo, a matriz “documento-tópico” pode ser calculada a partir do método de transformação.

    # LatentDirichletAllocation também implementa o método partial_fit. Isso é usado quando os dados podem ser buscados sequencialmente. 




    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py




    ## Referências:

    ## “Latent Dirichlet Allocation” D. Blei, A. Ng, M. Jordan, 2003 (http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

    ## “Online Learning for Latent Dirichlet Allocation” M. Hoffman, D. Blei, F. Bach, 2010 (https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf)

    ## “Stochastic Variational Inference” M. Hoffman, D. Blei, C. Wang, J. Paisley, 2013 (http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf)

    ## “The varimax criterion for analytic rotation in factor analysis” H. F. Kaiser, 1958 (https://link.springer.com/article/10.1007%2FBF02289233)