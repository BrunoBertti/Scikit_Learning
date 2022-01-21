########## 1.5.9. Detalhes de implementação  ##########

    # A implementação do SGD é influenciada pelo Gradiente Estocástico SVM de 7. Semelhante ao SvmSGD, o vetor de peso é representado como o produto de um escalar e um vetor que permite uma atualização de peso eficiente no caso de regularização L2. No caso da entrada esparsa X, o intercepto é atualizado com uma taxa de aprendizado menor (multiplicado por 0,01) para levar em conta o fato de ser atualizado com mais frequência. Os exemplos de treinamento são selecionados sequencialmente e a taxa de aprendizado é reduzida após cada exemplo observado. Adotamos o cronograma de taxa de aprendizado de 8. Para classificação multiclasse, é usada uma abordagem “um contra todos”. Usamos o algoritmo de gradiente truncado proposto em 9 para regularização L1 (e a Elastic Net). O código está escrito em Cython. 


    ## Referências:

    ## 7 “Stochastic Gradient Descent” L. Bottou - Website, 2010. (https://leon.bottou.org/projects/sgd)

    ## 8 “Pegasos: Primal estimated sub-gradient solver for svm” S. Shalev-Shwartz, Y. Singer, N. Srebro - In Proceedings of ICML ‘07. (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513)

    ## 9 “Stochastic gradient descent training for l1-regularized log-linear models with cumulative penalty” Y. Tsuruoka, J. Tsujii, S. Ananiadou - In Proceedings of the AFNLP/ACL ‘09. (https://www.aclweb.org/anthology/P/P09/P09-1054.pdf)

    ## 10(1,2) “Towards Optimal One Pass Large Scale Learning with Averaged Stochastic Gradient Descent” Xu, Wei (https://arxiv.org/pdf/1107.2490v2.pdf)

    ## 11 “Regularization and variable selection via the elastic net” H. Zou, T. Hastie - Journal of the Royal Statistical Society Series B, 67 (2), 301-320. (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696)

    ## 12 “Solving large scale linear prediction problems using stochastic gradient descent algorithms” T. Zhang - In Proceedings of ICML ‘04. (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.58.7377)