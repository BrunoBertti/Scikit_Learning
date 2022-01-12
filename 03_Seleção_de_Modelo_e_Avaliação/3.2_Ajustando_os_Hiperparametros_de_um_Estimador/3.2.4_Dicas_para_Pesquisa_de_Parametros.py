########## 3.2.4. Dicas para pesquisa de parâmetros ##########



##### 3.2.4.1. Como especificar uma métrica objetiva

    # Por padrão, a pesquisa de parâmetro usa a função de pontuação do estimador para avaliar uma configuração de parâmetro. Estes são sklearn.metrics.accuracy_score para classificação e sklearn.metrics.r2_score para regressão. Para algumas aplicações, outras funções de pontuação são mais adequadas (por exemplo, na classificação desequilibrada, a pontuação de precisão geralmente não é informativa). Uma função de pontuação alternativa pode ser especificada por meio do parâmetro de pontuação da maioria das ferramentas de pesquisa de parâmetros. Consulte O parâmetro de pontuação: definindo regras de avaliação de modelo para obter mais detalhes. 




##### 3.2.4.2. Especificando várias métricas para avaliação

    # GridSearchCV e RandomizedSearchCV permitem especificar várias métricas para o parâmetro de pontuação.

    # A pontuação multimétrica pode ser especificada como uma lista de sequências de nomes de pontuações predefinidos ou um dict mapeando o nome do pontuador para a função do pontuador e/ou o(s) nome(s) do pontuador predefinido. Consulte Usando a avaliação de várias métricas para obter mais detalhes.

    # Ao especificar várias métricas, o parâmetro refit deve ser definido para a métrica (string) para a qual o best_params_ será encontrado e usado para criar o best_estimator_ em todo o conjunto de dados. Se a pesquisa não deve ser reajustada, defina refit=False. Deixar o reajuste com o valor padrão Nenhum resultará em um erro ao usar várias métricas.

    # Consulte Demonstração de avaliação multimétrica em cross_val_score e GridSearchCV para obter um exemplo de uso.

    # HalvingRandomSearchCV e HalvingGridSearchCV não suportam pontuação multimétrica. 




##### 3.2.4.3. Estimadores compostos e espaços de parâmetros

    # GridSearchCV e RandomizedSearchCV permitem pesquisar parâmetros de estimadores compostos ou aninhados, como Pipeline, ColumnTransformer, VotingClassifier ou CalibratedClassifierCV usando uma sintaxe dedicada <estimator>__<parameter>: 

from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
X, y = make_moons()
calibrated_forest = CalibratedClassifierCV(
   base_estimator=RandomForestClassifier(n_estimators=10))
param_grid = {
   'base_estimator__max_depth': [2, 4, 6, 8]}
search = GridSearchCV(calibrated_forest, param_grid, cv=5)
search.fit(X, y)
GridSearchCV(cv=5,
             estimator=CalibratedClassifierCV(...),
             param_grid={'base_estimator__max_depth': [2, 4, 6, 8]})


    # Aqui, <estimator> é o nome do parâmetro do estimador aninhado, neste caso base_estimator. Se o meta-estimador for construído como uma coleção de estimadores como em pipeline.Pipeline, então <estimator> se refere ao nome do estimador, consulte Parâmetros aninhados. Na prática, pode haver vários níveis de aninhamento: 

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
pipe = Pipeline([
   ('select', SelectKBest()),
   ('model', calibrated_forest)])
param_grid = {
   'select__k': [1, 2],
   'model__base_estimator__max_depth': [2, 4, 6, 8]}
search = GridSearchCV(pipe, param_grid, cv=5).fit(X, y)


    # Consulte Pipeline: estimadores de encadeamento para realizar pesquisas de parâmetros em pipelines.


##### 3.2.4.4. Seleção de modelo: desenvolvimento e avaliação

    # A seleção do modelo avaliando várias configurações de parâmetros pode ser vista como uma forma de usar os dados rotulados para “treinar” os parâmetros da grade.

    # Ao avaliar o modelo resultante, é importante fazê-lo em amostras retidas que não foram vistas durante o processo de pesquisa da grade: recomenda-se dividir os dados em um conjunto de desenvolvimento (a ser alimentado na instância GridSearchCV) e um conjunto de avaliação para calcular métricas de desempenho.

    # Isso pode ser feito usando a função de utilitário train_test_split. 


##### 3.2.4.5. Paralelismo

    # As ferramentas de pesquisa de parâmetros avaliam cada combinação de parâmetros em cada dobra de dados de forma independente. Os cálculos podem ser executados em paralelo usando a palavra-chave n_jobs=-1. Consulte a assinatura da função para obter mais detalhes e também a entrada do Glossário para n_jobs. 



##### 3.2.4.6. Robustez ao fracasso 

    # Algumas configurações de parâmetros podem resultar em uma falha no ajuste de uma ou mais dobras dos dados. Por padrão, isso fará com que toda a pesquisa falhe, mesmo que algumas configurações de parâmetros possam ser totalmente avaliadas. Definir error_score=0 (ou =np.NaN) tornará o procedimento robusto a tal falha, emitindo um aviso e definindo o score dessa dobra para 0 (ou NaN), mas completando a busca. 

