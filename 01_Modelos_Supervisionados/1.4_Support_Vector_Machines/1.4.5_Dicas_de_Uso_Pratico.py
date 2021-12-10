########## 1.4.5 Dicas de Uso Prático ##########

    # Evitando cópia de dados: Para SVC, SVR, NuSVC e NuSVR, se os dados passados para certos métodos não forem ordenados em C contíguos e com precisão dupla, eles serão copiados antes de chamar a implementação C subjacente. Você pode verificar se um determinado array numpy é C-contíguo inspecionando seu atributo flags. 
    # Para LinearSVC (e LogisticRegression), qualquer entrada passada como uma matriz numpy será copiada e convertida para a representação de dados esparsos internos liblinear (flutuações de precisão dupla e índices int32 de componentes diferentes de zero). Se você deseja ajustar um classificador linear de grande escala sem copiar uma matriz densa numpy C-contígua de precisão dupla como entrada, sugerimos usar a classe SGDClassifier. A função objetivo pode ser configurada para ser quase a mesma do modelo LinearSVC. 

    # Tamanho do cache do kernel: Para SVC, SVR, NuSVC e NuSVR, o tamanho do cache do kernel tem um forte impacto nos tempos de execução para problemas maiores. Se você tiver RAM suficiente disponível, é recomendado definir cache_size para um valor maior do que o padrão de 200 (MB), como 500 (MB) ou 1000 (MB). 

    # Configuração C: C é 1 por padrão e é uma escolha padrão razoável. Se você tiver muitas observações com ruído, deve diminuí-las: diminuir C corresponde a mais regularização. 
    # LinearSVC e LinearSVR são menos sensíveis a C quando se torna grande e os resultados de predição param de melhorar após um certo limite. Enquanto isso, valores de C maiores levarão mais tempo para treinar, às vezes até 10 vezes mais, conforme mostrado em 11. 

    # Os algoritmos do Support Vector Machine não são invariáveis de escala, por isso é altamente recomendável dimensionar seus dados. Por exemplo, dimensione cada atributo no vetor de entrada X para [0,1] ou [-1, + 1], ou padronize-o para ter média 0 e variância 1. Observe que a mesma escala deve ser aplicada ao vetor de teste para obter resultados significativos. Isso pode ser feito facilmente usando um Pipeline: 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC())

    # Consulte a seção Dados de pré-processamento para obter mais detalhes sobre dimensionamento e normalização. 

    # Em relação ao parâmetro de redução, citando 12: Descobrimos que se o número de iterações for grande, a redução pode encurtar o tempo de treinamento. No entanto, se resolvermos vagamente o problema de otimização (por exemplo, usando uma grande tolerância de parada), o código sem usar a redução pode ser muito mais rápido

    # O parâmetro nu em NuSVC / OneClassSVM / NuSVR aproxima a fração de erros de treinamento e vetores de suporte.

    # No SVC, se os dados estiverem desequilibrados (por exemplo, muitos positivos e poucos negativos), defina class_weight = 'equilibrado' e / ou tente diferentes parâmetros de penalidade C. 

    # Aleatoriedade das implementações subjacentes: As implementações subjacentes de SVC e NuSVC usam um gerador de número aleatório apenas para embaralhar os dados para estimativa de probabilidade (quando a probabilidade é definida como Verdadeiro). Essa aleatoriedade pode ser controlada com o parâmetro random_state. Se a probabilidade for definida como False, esses estimadores não são aleatórios e random_state não tem efeito nos resultados. A implementação OneClassSVM subjacente é semelhante às de SVC e NuSVC. Como nenhuma estimativa de probabilidade é fornecida para OneClassSVM, ela não é aleatória.
    # A implementação LinearSVC subjacente usa um gerador de número aleatório para selecionar recursos ao ajustar o modelo com uma descida de coordenada dupla (ou seja, quando dual é definido como True). Portanto, não é incomum ter resultados ligeiramente diferentes para os mesmos dados de entrada. Se isso acontecer, tente com um parâmetro tol menor. Essa aleatoriedade também pode ser controlada com o parâmetro random_state. Quando dual é definido como False, a implementação subjacente de LinearSVC não é aleatória e random_state não tem efeito nos resultados. 

    # Usar a penalização L1 conforme fornecido pelo LinearSVC (penalidade = 'l1', dual = False) produz uma solução esparsa, ou seja, apenas um subconjunto de pesos de recursos é diferente de zero e contribui para a função de decisão. Aumentar C produz um modelo mais complexo (mais recursos são selecionados). O valor C que produz um modelo “nulo” (todos os pesos iguais a zero) pode ser calculado usando l1_min_c. 