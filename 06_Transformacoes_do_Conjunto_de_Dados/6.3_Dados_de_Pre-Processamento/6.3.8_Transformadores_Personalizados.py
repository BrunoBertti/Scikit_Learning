########## 6.3.8. Transformadores personalizados  ##########


    # Frequentemente, você desejará converter uma função Python existente em um transformador para auxiliar na limpeza ou processamento de dados. Você pode implementar um transformador de uma função arbitrária com FunctionTransformer. Por exemplo, para construir um transformador que aplique uma transformação de log em um pipeline, faça: 

import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p, validate=True)
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)

    # Você pode garantir que func e inverse_func sejam o inverso um do outro definindo check_inverse=True e chamando fit antes de transform. Observe que um aviso é gerado e pode ser transformado em erro com avisos de filtro: 

import warnings
warnings.filterwarnings("error", message=".*check_inverse*.",
                        category=UserWarning, append=False)


    # Para obter um exemplo de código completo que demonstra o uso de um FunctionTransformer para extrair recursos de dados de texto, consulte Column Transformer com fontes de dados heterogêneas e engenharia de recursos relacionados ao tempo. 