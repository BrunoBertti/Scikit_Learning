########## 6.1. Pipelines e estimadores compostos ##########



    # Os transformadores são geralmente combinados com classificadores, regressores ou outros estimadores para construir um estimador composto. A ferramenta mais comum é um Pipeline. Pipeline é frequentemente usado em combinação com FeatureUnion, que concatena a saída de transformadores em um espaço de recursos composto. TransformedTargetRegressor lida com a transformação do alvo (ou seja, log-transform y). Em contraste, Pipelines apenas transformam os dados observados (X). 

