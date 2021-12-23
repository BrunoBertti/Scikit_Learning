########## 1.14.1. Autotreinamento ##########


    # Esta implementação de autotreinamento é baseada no algoritmo 1 de Yarowsky. Usando esse algoritmo, um determinado classificador supervisionado pode funcionar como um classificador semissupervisionado, permitindo que ele aprenda a partir de dados não rotulados.

    # SelfTrainingClassifier pode ser chamado com qualquer classificador que implemente Predict_proba, passado como o parâmetro base_classifier. Em cada iteração, o base_classifier prevê rótulos para as amostras não rotuladas e adiciona um subconjunto desses rótulos ao conjunto de dados rotulado.

    # A escolha deste subconjunto é determinada pelo critério de seleção. Essa seleção pode ser feita usando um limite nas probabilidades de previsão ou escolhendo as amostras k_best de acordo com as probabilidades de previsão.

    # Os rótulos usados ​​para o ajuste final, bem como a iteração em que cada amostra foi rotulada, estão disponíveis como atributos. O parâmetro opcional max_iter especifica quantas vezes o loop é executado no máximo.

    # O parâmetro max_iter pode ser definido como Nenhum, fazendo com que o algoritmo itere até que todas as amostras tenham rótulos ou nenhuma nova amostra seja selecionada nessa iteração. 

    # Nota: Ao usar o classificador de autotreinamento, a calibração do classificador é importante. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_self_training_varying_threshold.html#sphx-glr-auto-examples-semi-supervised-plot-self-training-varying-threshold-py

    ## https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_semi_supervised_versus_svm_iris.html#sphx-glr-auto-examples-semi-supervised-plot-semi-supervised-versus-svm-iris-py




    ## Referências:

    ##  David Yarowsky. 1995. Unsupervised word sense disambiguation rivaling supervised methods. In Proceedings of the 33rd annual meeting on Association for Computational Linguistics (ACL ‘95). Association for Computational Linguistics, Stroudsburg, PA, USA, 189-196. DOI: https://doi.org/10.3115/981658.981684 (https://doi.org/10.3115/981658.981684)