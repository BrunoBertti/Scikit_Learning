########## 6.4.3. Imputação de recurso multivariável ##########


    # Uma abordagem mais sofisticada é usar a classe IterativeImputer, que modela cada recurso com valores ausentes em função de outros recursos e usa essa estimativa para imputação. Ele faz isso de forma iterativa round-robin: em cada etapa, uma coluna de recurso é designada como saída y e as outras colunas de recurso são tratadas como entradas X. Um regressor é ajustado em (X, y) para y conhecido. Em seguida, o regressor é usado para prever os valores ausentes de y. Isso é feito para cada recurso de maneira iterativa e, em seguida, repetido para rodadas de imputação max_iter. Os resultados da rodada de imputação final são retornados.

    # Nota: Este estimador ainda é experimental por enquanto: parâmetros padrão ou detalhes de comportamento podem mudar sem qualquer ciclo de depreciação. Resolver os seguintes problemas ajudaria a estabilizar IterativeImputer: critérios de convergência (#14338), estimadores padrão (#13286) e uso de estado aleatório (#15611). Para usá-lo, você precisa importar explicitamente enable_iterative_imputer. 

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])
IterativeImputer(random_state=0)
X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
# o modelo aprende que o segundo recurso é o dobro do primeiro 
print(np.round(imp.transform(X_test)))


    # Tanto SimpleImputer quanto IterativeImputer podem ser usados em um Pipeline como uma forma de construir um estimador composto que suporte a imputação. Consulte Impor valores ausentes antes de construir um estimador. 



##### 6.4.3.1. Flexibilidade do IterativeImputer

    # Existem muitos pacotes de imputação bem estabelecidos no ecossistema de ciência de dados R: Amelia, mi, mouse, missForest, etc. missForest é popular e acaba sendo uma instância particular de diferentes algoritmos de imputação sequencial que podem ser implementados com IterativeImputer por passando em diferentes regressores a serem usados para prever valores de recursos ausentes. No caso de missForest, esse regressor é uma Random Forest. Consulte Imputando valores ausentes com variantes de IterativeImputer. 








##### 6.4.3.2. Imputação múltipla vs. única 


    # Na comunidade estatística, é prática comum realizar múltiplas imputações, gerando, por exemplo, m imputações separadas para uma única matriz de características. Cada uma dessas m imputações é então colocada no pipeline de análise subsequente (por exemplo, engenharia de recursos, agrupamento, regressão, classificação). Os m resultados da análise final (por exemplo, erros de validação retidos) permitem que o cientista de dados compreenda como os resultados analíticos podem diferir como consequência da incerteza inerente causada pelos valores ausentes. A prática acima é chamada de imputação múltipla.

    # Nossa implementação do IterativeImputer foi inspirada no pacote R MICE (Multivariate Imputation by Chained Equations) 1, mas difere dele por retornar uma única imputação ao invés de múltiplas imputações. No entanto, IterativeImputer também pode ser usado para múltiplas imputações aplicando-o repetidamente ao mesmo conjunto de dados com diferentes sementes aleatórias quando sample_posterior=True. Ver 2, capítulo 4 para mais discussão sobre imputações múltiplas vs. simples.

    # Ainda é um problema em aberto a utilidade da imputação simples versus múltipla no contexto de previsão e classificação quando o usuário não está interessado em medir a incerteza devido a valores ausentes.

    # Observe que uma chamada para o método transform de IterativeImputer não tem permissão para alterar o número de amostras. Portanto, múltiplas imputações não podem ser alcançadas por uma única chamada para transformar. 