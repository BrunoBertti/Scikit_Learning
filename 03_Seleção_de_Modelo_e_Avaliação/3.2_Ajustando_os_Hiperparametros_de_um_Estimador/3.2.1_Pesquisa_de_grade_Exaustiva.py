########## 3.2.1. Pesquisa de grade exaustiva ##########

    # A pesquisa de grade fornecida pelo GridSearchCV gera exaustivamente candidatos a partir de uma grade de valores de parâmetro especificados com o parâmetro param_grid. Por exemplo, o seguinte param_grid: 

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]


    # especifica que duas grades devem ser exploradas: uma com um kernel linear e valores C em [1, 10, 100, 1000], e a segunda com um kernel RBF, e o produto cruzado de valores C variando em [1, 10 , 100, 1000] e valores gama em [0,001, 0,0001].

    # A instância GridSearchCV implementa a API do estimador usual: ao “ajustá-la” em um conjunto de dados, todas as combinações possíveis de valores de parâmetros são avaliadas e a melhor combinação é mantida. 



    ## Exemplos:

    ## See Parameter estimation using grid search with cross-validation for an example of Grid Search computation on the digits dataset. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py)

    ## See Sample pipeline for text feature extraction and evaluation for an example of Grid Search coupling parameters from a text documents feature extractor (n-gram count vectorizer and TF-IDF transformer) with a classifier (here a linear SVM trained with SGD with either elastic net or L2 penalty) using a pipeline.Pipeline instance. (https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py)

    ## See Nested versus non-nested cross-validation for an example of Grid Search within a cross validation loop on the iris dataset. This is the best practice for evaluating the performance of a model with grid search. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py)

    ## See Demonstration of multi-metric evaluation on cross_val_score and GridSearchCV for an example of GridSearchCV being used to evaluate multiple metrics simultaneously. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py)

    ## See Balance model complexity and cross-validated score for an example of using refit=callable interface in GridSearchCV. The example shows how this interface adds certain amount of flexibility in identifying the “best” estimator. This interface can also be used in multiple metrics evaluation. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html#sphx-glr-auto-examples-model-selection-plot-grid-search-refit-callable-py)

    ## See Statistical comparison of models using grid search for an example of how to do a statistical comparison on the outputs of GridSearchCV. (https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py)