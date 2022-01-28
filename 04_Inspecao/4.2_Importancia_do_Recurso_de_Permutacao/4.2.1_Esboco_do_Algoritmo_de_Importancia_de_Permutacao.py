########## 4.2.1. Esboço do algoritmo de importância de permutação ##########

    # Entradas: modelo preditivo ajustado m, conjunto de dados tabular (treinamento ou validação) D.

    # Calcule a pontuação de referência s do modelo m nos dados D (por exemplo, a precisão para um classificador ou o R^2 para um regressor).

    # Para cada recurso j (coluna de D):

        # Para cada repetição k em {1, ..., K}:

            # Embaralhe aleatoriamente a coluna j do conjunto de dados D para gerar uma versão corrompida dos dados chamada \tilde{D}_{k,j}.

            # Calcule a pontuação s_{k,j} do modelo j em dados corrompidos \tilde{D}_{k,j}.

        # Calcule a importância i_j para o recurso f_j definido como: 