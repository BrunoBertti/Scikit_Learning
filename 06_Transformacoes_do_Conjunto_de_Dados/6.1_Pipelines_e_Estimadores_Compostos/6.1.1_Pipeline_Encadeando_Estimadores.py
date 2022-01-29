########## 6.1.1. Pipeline: encadeando estimadores ##########



    # O pipeline pode ser usado para encadear vários estimadores em um. Isso é útil, pois geralmente há uma sequência fixa de etapas no processamento dos dados, por exemplo, seleção de recursos, normalização e classificação. O pipeline serve a vários propósitos aqui:

        # Comodidade e encapsulamento
            # Você só precisa chamar ajuste e prever uma vez em seus dados para ajustar toda uma sequência de estimadores.

        # Seleção de parâmetros conjuntos
            # Você pode pesquisar em grade os parâmetros de todos os estimadores no pipeline de uma só vez.

        # Segurança
            # Os pipelines ajudam a evitar o vazamento de estatísticas de seus dados de teste no modelo treinado na validação cruzada, garantindo que as mesmas amostras sejam usadas para treinar os transformadores e preditores.

    # Todos os estimadores em um pipeline, exceto o último, devem ser transformadores (ou seja, devem ter um método de transformação). O último estimador pode ser de qualquer tipo (transformador, classificador, etc.). 




##### 6.1.1.1. Uso


##### 6.1.1.1.1. Construção

    # O Pipeline é construído usando uma lista de pares (chave, valor), onde a chave é uma string contendo o nome que você deseja dar a esta etapa e valor é um objeto estimador: 

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)
pipe

    # A função utilitária make_pipeline é uma abreviação para construir pipelines; ele pega um número variável de estimadores e retorna um pipeline, preenchendo os nomes automaticamente: 

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
make_pipeline(Binarizer(), MultinomialNB())



##### 6.1.1.1.2. Etapas de acesso

    # Os estimadores de um pipeline são armazenados como uma lista no atributo steps, mas podem ser acessados por índice ou nome indexando (com [idx]) o Pipeline: 

pipe.steps[0]
('reduce_dim', PCA())
pipe[0]
pipe['reduce_dim']

    # O atributo named_steps do pipeline permite acessar as etapas por nome com preenchimento de guias em ambientes interativos: 

pipe.named_steps.reduce_dim is pipe['reduce_dim']


    # Um sub-pipeline também pode ser extraído usando a notação de fatiamento comumente usada para sequências Python, como listas ou strings (embora apenas uma etapa de 1 seja permitida). Isso é conveniente para realizar apenas algumas das transformações (ou sua inversa): 

pipe[:1]
Pipeline(steps=[('reduce_dim', PCA())])
pipe[-1:]




##### 6.1.1.1.3. Parâmetros aninhados

    # Os parâmetros dos estimadores no pipeline podem ser acessados usando a sintaxe <estimator>__<parameter>: 

pipe.set_params(clf__C=10)


    # Isso é particularmente importante para fazer pesquisas de grade: 

from sklearn.model_selection import GridSearchCV
param_grid = dict(reduce_dim__n_components=[2, 5, 10],
                  clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)

    # Etapas individuais também podem ser substituídas como parâmetros, e etapas não finais podem ser ignoradas definindo-as como 'passthrough': 

from sklearn.linear_model import LogisticRegression
param_grid = dict(reduce_dim=['passthrough', PCA(5), PCA(10)],
                  clf=[SVC(), LogisticRegression()],
                  clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)

    # Os estimadores do pipeline podem ser recuperados por índice: 


pipe[0]

    # ou pelo nome: 

pipe['reduce_dim']


    # Para habilitar a inspeção do modelo, o Pipeline possui um método get_feature_names_out(), assim como todos os transformadores. Você pode usar o fatiamento de pipeline para obter os nomes dos recursos em cada etapa: 

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
iris = load_iris()
pipe = Pipeline(steps=[
   ('select', SelectKBest(k=2)),
   ('clf', LogisticRegression())])
pipe.fit(iris.data, iris.target)
Pipeline(steps=[('select', SelectKBest(...)), ('clf', LogisticRegression(...))])
pipe[:-1].get_feature_names_out()

    # Você também pode fornecer nomes de recursos personalizados para os dados de entrada usando get_feature_names_out: 

pipe[:-1].get_feature_names_out(iris.feature_names)



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection_pipeline.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-pipeline-py

    ## https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

    ## https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html#sphx-glr-auto-examples-compose-plot-digits-pipe-py

    ## https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html#sphx-glr-auto-examples-miscellaneous-plot-kernel-approximation-py

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_svm_anova.html#sphx-glr-auto-examples-svm-plot-svm-anova-py

    ## https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py

    ## https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_pipeline_display.html#sphx-glr-auto-examples-miscellaneous-plot-pipeline-display-py



    ## Ver Também:

    ## https://scikit-learn.org/stable/modules/grid_search.html#composite-grid-search









##### 6.1.1.2. Notas

    # Chamar fit no pipeline é o mesmo que chamar fit em cada estimador, transformar a entrada e passá-la para a próxima etapa. O pipeline possui todos os métodos que o último estimador do pipeline possui, ou seja, se o último estimador for um classificador, o Pipeline pode ser usado como classificador. Se o último estimador for um transformador, novamente, o pipeline também será. 



##### 6.1.1.3. Transformadores de cache: evite computação repetida 

    # Adaptar transformadores pode ser computacionalmente caro. Com seu parâmetro de memória definido, o Pipeline armazenará em cache cada transformador após chamar fit. Esse recurso é usado para evitar calcular os transformadores de ajuste em um pipeline se os parâmetros e os dados de entrada forem idênticos. Um exemplo típico é o caso de uma pesquisa de rede em que os transformadores podem ser instalados apenas uma vez e reutilizados para cada configuração.

    # A memória de parâmetros é necessária para armazenar em cache os transformadores. memory pode ser uma string contendo o diretório onde armazenar em cache os transformadores ou um objeto joblib.Memory: 

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
cachedir = mkdtemp()
pipe = Pipeline(estimators, memory=cachedir)
pipe
Pipeline(memory=...,
         steps=[('reduce_dim', PCA()), ('clf', SVC())])
# Limpe o diretório de cache quando não precisar mais dele 
rmtree(cachedir)


    # Aviso: efeito colateral de transformadores de cache
    # Usando um Pipeline sem cache habilitado, é possível inspecionar a instância original como: 


from sklearn.datasets import load_digits
X_digits, y_digits = load_digits(return_X_y=True)
pca1 = PCA()
svm1 = SVC()
pipe = Pipeline([('reduce_dim', pca1), ('clf', svm1)])
pipe.fit(X_digits, y_digits)
Pipeline(steps=[('reduce_dim', PCA()), ('clf', SVC())])
# A instância pca pode ser inspecionada diretamente
print(pca1.components_)   

    # Habilitar o cache aciona um clone dos transformadores antes do ajuste. Portanto, a instância do transformador fornecida ao pipeline não pode ser inspecionada diretamente. No exemplo a seguir, acessar a instância do PCA pca2 gerará um AttributeError, pois pca2 será um transformador não ajustado. Em vez disso, use o atributo named_steps para inspecionar os estimadores no pipeline: 

cachedir = mkdtemp()
pca2 = PCA()
svm2 = SVC()
cached_pipe = Pipeline([('reduce_dim', pca2), ('clf', svm2)],
                       memory=cachedir)
cached_pipe.fit(X_digits, y_digits)
Pipeline(memory=...,
        steps=[('reduce_dim', PCA()), ('clf', SVC())])
print(cached_pipe.named_steps['reduce_dim'].components_)
# Remova o diretório de cache 
rmtree(cachedir)



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py