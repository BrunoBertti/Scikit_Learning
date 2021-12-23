##########  1.13.6. Seleção de variáveis como parte de um pipeline  ##########


    # A seleção de variáveis geralmente é usada como uma etapa de pré-processamento antes de fazer o aprendizado real. A maneira recomendada de fazer isso no scikit-learn é usar um Pipeline: 


from sklearn.pipeline import Pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)


    # Neste trecho, usamos um LinearSVC acoplado a SelectFromModel para avaliar as importâncias do recurso e selecionar os recursos mais relevantes. Em seguida, um RandomForestClassifier é treinado na saída transformada, ou seja, usando apenas recursos relevantes. Você pode realizar operações semelhantes com os outros métodos de seleção de recursos e também classificadores que fornecem uma maneira de avaliar as importâncias dos recursos, é claro. Veja os exemplos de Pipeline para mais detalhes. 