########## 6.7.1. Método Nystroem para Aproximação do Kernel ##########


    # O método Nystroem, conforme implementado no Nystroem, é um método geral para aproximações de núcleos de baixo nível. Ele consegue isso essencialmente subamostrando os dados nos quais o kernel é avaliado. Por padrão, o Nystroem usa o kernel rbf, mas pode usar qualquer função do kernel ou uma matriz de kernel pré-computada. O número de amostras utilizadas - que também é a dimensionalidade dos recursos computados - é dado pelo parâmetro n_components. 