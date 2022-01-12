########## 3.3.3. Métricas de classificação de vários rótulos ##########

    # No aprendizado multirrótulo, cada amostra pode ter qualquer número de rótulos de verdade associados a ela. O objetivo é dar pontuações mais altas e melhor classificação aos rótulos de verdade. 


#####  3.3.3.1. Erro de cobertura

    # A função coverage_error calcula o número médio de rótulos que devem ser incluídos na previsão final, de modo que todos os rótulos verdadeiros sejam previstos. Isso é útil se você quiser saber quantos rótulos com as melhores pontuações você precisa prever em média sem perder nenhum verdadeiro. O melhor valor dessa métrica é, portanto, o número médio de rótulos verdadeiros.

    # Nota: A pontuação da nossa implementação é 1 a mais do que a dada em Tsoukas et al., 2010. Isso a estende para lidar com o caso degenerado em que uma instância tem 0 rótulos verdadeiros.

    # Formalmente, dada uma matriz de indicadores binários dos rótulos de verdade do terreno y \in \left\{0, 1\right\}^{n_\text{samples} \times n_\text{labels}} e a pontuação associada a cada rótulo \hat{f}\inmathbb{R}^{n_\text{samples} \times n_\text{labels}}, a cobertura é definida como


        #cobertura(y, \hat{f}) = \frac{1}{n_{\text{amostras}}}
        #   \sum_{i=0}^{n_{\text{samples}} - 1} \max_{j:y_{ij} = 1} \text{rank}_{ij}


    # com \text{rank}_{ij} = \left|\left\{k: \hat{f}_{ik} \geq \hat{f}_{ij} \right\}\right|. Dada a definição de classificação, os empates em y_scores são quebrados fornecendo a classificação máxima que teria sido atribuída a todos os valores vinculados.

    # Aqui está um pequeno exemplo de uso desta função: 


import numpy as np
from sklearn.metrics import coverage_error
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
coverage_error(y_true, y_score)

#####  3.3.3.2. Precisão média de classificação do rótulo


    # A função label_ranking_average_precision_score implementa a precisão média de classificação de rótulo (LRAP). Essa métrica está vinculada à função average_precision_score, mas é baseada na noção de classificação de rótulo em vez de precisão e recall.

    # A precisão média de classificação de rótulos (LRAP) calcula as médias sobre as amostras a resposta para a seguinte pergunta: para cada rótulo de verdade, que fração dos rótulos de classificação mais alta eram rótulos verdadeiros? Essa medida de desempenho será maior se você conseguir classificar melhor os rótulos associados a cada amostra. A pontuação obtida é sempre estritamente maior que 0, e o melhor valor é 1. Se houver exatamente um rótulo relevante por amostra, a precisão média da classificação do rótulo é equivalente à classificação recíproca média.

    # Formalmente, dada uma matriz de indicadores binários dos rótulos de verdade do terreno y \in \left\{0, 1\right\}^{n_\text{samples} \times n_\text{labels}} e a pontuação associada a cada rótulo \hat{f} \in \mathbb{R}^{n_\text{samples} \times n_\text{labels}}, a precisão média é definida como 

        # LRAP(y, \hat{f}) = \frac{1}{n_{\text{amostras}}}
        #    \sum_{i=0}^{n_{\text{amostras}} - 1} \frac{1}{||y_i||_0}
        #    \sum_{j:y_{ij} = 1} \frac{|\mathcal{L}_{ij}|}{\text{rank}_{ij}}

    # onde \mathcal{L}_{ij} = \left\{k: y_{ik} = 1, \hat{f}_{ik} \geq \hat{f}_{ij} \right\}, \ text{rank}_{ij} = \left|\left\{k: \hat{f}_{ik} \geq \hat{f}_{ij} \right\}\right|, calcula a cardinalidade de o conjunto (ou seja, o número de elementos no conjunto), e ||\cdot||_0 é a \ell_0 “norma” (que calcula o número de elementos diferentes de zero em um vetor).



    #Aqui está um pequeno exemplo de uso desta função: 

import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
label_ranking_average_precision_score(y_true, y_score)   



#####  3.3.3.3. Perda de classificação

    # A função label_ranking_loss calcula a perda de classificação que calcula sobre as amostras o número de pares de rótulos que estão ordenados incorretamente, ou seja, rótulos verdadeiros têm uma pontuação menor do que rótulos falsos, ponderados pelo inverso do número de pares ordenados de rótulos falsos e verdadeiros. A menor perda de classificação alcançável é zero.

    # Formalmente, dada uma matriz de indicadores binários dos rótulos de verdade do terreno y \in \left\{0, 1\right\}^{n_\text{samples} \times n_\text{labels}} e a pontuação associada a cada rótulo \hat{f} \in \mathbb{R}^{n_\text{samples} \times n_\text{labels}}, a perda de classificação é definida como


        # ranking\_loss(y, \hat{f}) = \frac{1}{n_{\text{amostras}}}
        #   \sum_{i=0}^{n_{\text{samples}} - 1} \frac{1}{||y_i||_0(n_\text{labels} - ||y_i||_0)}
        #   \left|\left\{(k, l): \hat{f}_{ik} \leq \hat{f}_{il}, y_{ik} = 1, y_{il} = 0 \right\ }\direito|


    # onde |\cdot| calcula a cardinalidade do conjunto (ou seja, o número de elementos no conjunto) e ||\cdot||_0 é a “norma” \ell_0 (que calcula o número de elementos diferentes de zero em um vetor).

    # Aqui está um pequeno exemplo de uso desta função: 


import numpy as np
from sklearn.metrics import label_ranking_loss
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
label_ranking_loss(y_true, y_score)
# Com a seguinte previsão, temos perda perfeita e mínima 
y_score = np.array([[1.0, 0.1, 0.2], [0.1, 0.2, 0.9]])
label_ranking_loss(y_true, y_score)


    ## Referências:

    ## Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and knowledge discovery handbook (pp. 667-685). Springer US.




#####  3.3.3.4. Ganho Acumulado Descontado Normalizado 

    # Ganho cumulativo descontado (DCG) e Ganho cumulativo descontado normalizado (NDCG) são métricas de classificação implementadas em dcg_score e ndcg_score ; eles comparam uma ordem prevista com pontuações de verdade, como a relevância das respostas para uma consulta.

    # Da página da Wikipedia para Ganho Cumulativo Descontado:

    # “O ganho cumulativo com desconto (DCG) é uma medida da qualidade do ranking. Na recuperação de informações, muitas vezes é usado para medir a eficácia dos algoritmos do mecanismo de pesquisa da Web ou aplicativos relacionados. Usando uma escala de relevância graduada de documentos em um conjunto de resultados de mecanismo de pesquisa, o DCG mede a utilidade ou ganho de um documento com base em sua posição na lista de resultados. O ganho é acumulado do topo da lista de resultados para o fundo, com o ganho de cada resultado descontado nas classificações mais baixas”

    # O DCG ordena os alvos verdadeiros (por exemplo, relevância das respostas da consulta) na ordem prevista, depois os multiplica por um decaimento logarítmico e soma o resultado. A soma pode ser truncada após os primeiros K resultados, caso em que a chamamos de DCG@K. NDCG, ou NDCG@K é DCG dividido pelo DCG obtido por uma predição perfeita, de modo que esteja sempre entre 0 e 1. Normalmente, NDCG é preferível ao DCG.

    # Comparado com a perda de classificação, o NDCG pode levar em conta pontuações de relevância, em vez de uma classificação de verdade. Portanto, se a verdade básica consiste apenas em uma ordenação, a perda de classificação deve ser preferida; se a verdade básica consiste em pontuações de utilidade real (por exemplo, 0 para irrelevante, 1 para relevante, 2 para muito relevante), o NDCG pode ser usado.

    # Para uma amostra, dado o vetor de valores contínuos de verdade para cada alvo y \in \mathbb{R}^{M}, onde M é o número de saídas, e a previsão \hat{y}, que induz a classificação função f, a pontuação do DCG é

        # \sum_{r=1}^{\min(K, M)}\frac{y_{f(r)}}{\log(1 + r)}

    # e o escore NDCG é o escore do DCG dividido pelo escore do DCG obtido para y. 



    ## Referências:

    ## Wikipedia entry for Discounted Cumulative Gain (https://en.wikipedia.org/wiki/Discounted_cumulative_gain)

    ## Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems (TOIS), 20(4), 422-446.

    ## Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T. Y. (2013, May). A theoretical analysis of NDCG ranking measures. In Proceedings of the 26th Annual Conference on Learning Theory (COLT 2013)

    ## McSherry, F., & Najork, M. (2008, March). Computing information retrieval performance measures efficiently in the presence of tied scores. In European conference on information retrieval (pp. 414-421). Springer, Berlin, Heidelberg.