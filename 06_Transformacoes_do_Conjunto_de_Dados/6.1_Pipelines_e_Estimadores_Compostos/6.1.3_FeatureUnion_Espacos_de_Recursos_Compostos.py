########## 6.1.3. FeatureUnion: espaços de recursos compostos ##########

    # FeatureUnion combina vários objetos de transformador em um novo transformador que combina sua saída. Um FeatureUnion pega uma lista de objetos transformadores. Durante o ajuste, cada um deles é ajustado aos dados independentemente. Os transformadores são aplicados em paralelo e as matrizes de recursos que eles produzem são concatenadas lado a lado em uma matriz maior.

    # Quando você deseja aplicar diferentes transformações a cada campo dos dados, consulte a classe relacionada ColumnTransformer (consulte o guia do usuário).

    # O FeatureUnion atende aos mesmos propósitos do Pipeline - conveniência e estimativa e validação de parâmetros conjuntos.

    # FeatureUnion e Pipeline podem ser combinados para criar modelos complexos.

    # (Um FeatureUnion não tem como verificar se dois transformadores podem produzir recursos idênticos. Ele só produz uma união quando os conjuntos de recursos são disjuntos, e certificar-se de que eles são responsabilidade do chamador.)



##### 6.1.3.1. Uso 

    # Um FeatureUnion é construído usando uma lista de pares (chave, valor), onde a chave é o nome que você deseja dar a uma determinada transformação (uma string arbitrária; ela serve apenas como um identificador) e o valor é um objeto estimador: 

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
combined = FeatureUnion(estimators)
combined

    # Assim como os pipelines, as uniões de recursos têm um construtor abreviado chamado make_union que não requer nomenclatura explícita dos componentes.

    # Como o Pipeline, etapas individuais podem ser substituídas usando set_params e ignoradas definindo como 'drop': 

combined.set_params(kernel_pca='drop')
FeatureUnion(transformer_list=[('linear_pca', PCA()),
                               ('kernel_pca', 'drop')])



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py