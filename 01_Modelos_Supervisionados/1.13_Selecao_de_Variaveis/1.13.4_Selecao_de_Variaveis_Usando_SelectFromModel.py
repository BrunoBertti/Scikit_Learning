########## 1.13.4. Seleção de variáveis usando SelectFromModel ##########

    # SelectFromModel é um meta-transformador que pode ser usado junto com qualquer estimador que atribua importância a cada recurso por meio de um atributo específico (como coef_, feature_importances_) ou por meio de umght_getter que pode ser chamado após o ajuste. Os recursos são considerados sem importância e removidos se a importância correspondente dos valores do recurso estiver abaixo do parâmetro de limite fornecido. Além de especificar o limite numericamente, existem heurísticas integradas para encontrar um limite usando um argumento de string. As heurísticas disponíveis são “média”, “mediana” e múltiplos flutuantes dessas como “0,1 * média”. Em combinação com os critérios de limite, pode-se usar o parâmetro max_features para definir um limite no número de recursos a serem selecionados.

    # Para exemplos de como deve ser usado, consulte as seções abaixo. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py



##### 1.13.4.1. Seleção de recursos baseada em L1 

    # Modelos lineares penalizados com a norma L1 têm soluções esparsas: muitos de seus coeficientes estimados são zero. Quando o objetivo é reduzir a dimensionalidade dos dados para usar com outro classificador, eles podem ser usados junto com SelectFromModel para selecionar os coeficientes diferentes de zero. Em particular, estimadores esparsos úteis para esta finalidade são o Lasso para regressão e LogisticRegression e LinearSVC para classificação: 

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
X, y = load_iris(return_X_y=True)
X.shape
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape

    # Com SVMs e regressão logística, o parâmetro C controla a esparsidade: quanto menor C, menos recursos selecionados. Com o Lasso, quanto mais alto o parâmetro alfa, menos recursos selecionados. 


    ## Exemplos:

    ## Comparison of different algorithms for document classification including L1-based feature selection. (https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py)


    # Recuperação L1 e detecção compressiva

    # Para uma boa escolha de alfa, o Lasso pode recuperar totalmente o conjunto exato de variáveis ​​diferentes de zero usando apenas algumas observações, desde que certas condições específicas sejam atendidas. Em particular, o número de amostras deve ser "suficientemente grande", ou os modelos L1 serão executados aleatoriamente, onde "suficientemente grande" depende do número de coeficientes diferentes de zero, o logaritmo do número de recursos, a quantidade de ruído, o menor valor absoluto de coeficientes diferentes de zero e a estrutura da matriz de design X. Além disso, a matriz de design deve exibir certas propriedades específicas, como não ser muito correlacionada.

    # Não existe uma regra geral para selecionar um parâmetro alfa para recuperação de coeficientes diferentes de zero. Ele pode ser definido por validação cruzada (LassoCV ou LassoLarsCV), embora isso possa levar a modelos sub-penalizados: incluir um pequeno número de variáveis ​​não relevantes não é prejudicial para a pontuação de predição. BIC (LassoLarsIC) tende, ao contrário, a definir altos valores de alfa.

    # Referência Richard G. Baraniuk “Compressive Sensing”, IEEE Signal Processing Magazine [120] Julho 2007 http://users.isr.ist.utl.pt/~aguiar/CS_notes.pdf 


##### 1.13.4.2. Seleção de variáveis baseada em árvore 

    # Estimadores baseados em árvore (consulte o módulo sklearn.tree e floresta de árvores no módulo sklearn.ensemble) podem ser usados para calcular importâncias de recursos baseados em impurezas, que por sua vez podem ser usados para descartar recursos irrelevantes (quando acoplados com o meta SelectFromModel -transformador): 

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
X, y = load_iris(return_X_y=True)
X.shape
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape  



    ## Exemplos:

    ## example on synthetic data showing the recovery of the actually meaningful features. (https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py)

    ## example on face recognition data (https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-faces-py)