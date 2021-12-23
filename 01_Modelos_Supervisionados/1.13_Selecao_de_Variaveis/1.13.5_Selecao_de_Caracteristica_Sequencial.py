########## 1.13.5. Seleção de Característica Sequencial ##########



    # Sequential Feature Selection [sfs] (SFS) está disponível no transformador SequentialFeatureSelector. SFS pode ser para frente ou para trás:

    # Forward-SFS é um procedimento ganancioso que encontra iterativamente o melhor novo variável para adicionar ao conjunto de variáveis selecionados. Concretamente, inicialmente começamos com o variável zero e encontramos aquele que maximiza uma pontuação de validação cruzada quando um estimador é treinado neste variável único. Uma vez que o primeiro variável é selecionado, repetimos o procedimento adicionando um novo variável ao conjunto de variáveis selecionados. O procedimento para quando o número desejado de variáveis selecionados é alcançado, conforme determinado pelo parâmetro n_features_to_select.

    # O Backward-SFS segue a mesma ideia, mas funciona na direção oposta: em vez de começar sem nenhum variável e adicionar variáveis avidamente, começamos com todos os variáveis e removemos avidamente variáveis do conjunto. O parâmetro de direção controla se o SFS para frente ou para trás é usado.

    # Em geral, a seleção para frente e para trás não produz resultados equivalentes. Além disso, um pode ser muito mais rápido do que o outro, dependendo do número solicitado de variáveis selecionados: se tivermos 10 variáveis e solicitarmos 7 variáveis selecionados, a seleção para frente precisaria realizar 7 iterações, enquanto a seleção para trás só precisaria realizar 3.

    # SFS difere de RFE e SelectFromModel porque não requer que o modelo subjacente exponha um atributo coef_ ou feature_importances_. No entanto, pode ser mais lento considerando que mais modelos precisam ser avaliados, em comparação com as outras abordagens. Por exemplo, na seleção reversa, a iteração indo de m variáveis para m - 1 variáveis usando validação cruzada k-fold requer o ajuste de modelos m * k, enquanto RFE exigiria apenas um único ajuste, e SelectFromModel sempre faz apenas um único ajuste e requer sem iterações. 


    ## exemplos:

    ## https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py


    ## Referências:

    ## http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.24.4369&rep=rep1&type=pdf
