########## 1.12.3. Classificação multiclasse-multi saída ##########

    # A classificação multiclasse-multioutput (também conhecida como classificação multitarefa) é uma tarefa de classificação que rotula cada amostra com um conjunto de propriedades não binárias. Tanto o número de propriedades quanto o número de classes por propriedade é maior do que 2. Um único estimador lida com várias tarefas de classificação conjunta. Esta é tanto uma generalização da tarefa de classificação multilabel, que considera apenas atributos binários, quanto uma generalização da tarefa de classificação multiclasse, onde apenas uma propriedade é considerada.

    # Por exemplo, classificação das propriedades “tipo de fruta” e “cor” para um conjunto de imagens de fruta. A propriedade “tipo de fruta” possui as classes possíveis: “maçã”, “pêra” e “laranja”. A propriedade “color” possui as classes possíveis: “green”, “red”, “yellow” e “orange”. Cada amostra é uma imagem de uma fruta, um rótulo é produzido para ambas as propriedades e cada rótulo é uma das classes possíveis da propriedade correspondente.

    # Observe que todos os classificadores que lidam com tarefas multiclasse-multi-saída (também conhecidas como classificação multitarefa) suportam a tarefa de classificação multilabel como um caso especial. A classificação multitarefa é semelhante à tarefa de classificação multitarefa com diferentes formulações de modelo. Para obter mais informações, consulte a documentação do estimador relevante.

    # Aviso: no momento, nenhuma métrica em sklearn.metrics oferece suporte à tarefa de classificação multiclasse-multioutput. 



##### 1.12.3.1. Formato de destino 

    # Uma representação válida de multioutput y é uma matriz densa de forma (n_samples, n_classes) de rótulos de classe. Uma concatenação inteligente de colunas de variáveis multiclasse 1d. Um exemplo de y para 3 amostras: 


import numpy as np
y = np.array([['apple', 'green'], ['orange', 'orange'], ['pear', 'green']])
print(y)