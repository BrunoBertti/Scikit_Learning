############ 1.8.1. PLSCanonical ############

    # Descrevemos aqui o algoritmo usado no PLSCanonical. Os outros estimadores usam variantes desse algoritmo e são detalhados a seguir. Recomendamos a seção 1 para obter mais detalhes e comparações entre esses algoritmos. Em 1, PLSCanonical corresponde a “PLSW2A”.

   # Dadas duas matrizes centradas X \in \mathbb{R}^{n \times d} e Y \in \mathbb{R}^{n \times t}, e um número de componentes K, PLSCanonical procede da seguinte forma:

    # Defina X_1 como X e Y_1 como Y. Então, para cada k \in [1, K]: 

        # a) calcule u_k \in \mathbb{R}^d e v_k \in \mathbb{R}^t, os primeiros vetores singulares esquerdo e direito da matriz de covariância cruzada C = X_k^T Y_k u_k e v_k são chamados de pesos. Por definição, u_k e v_k são escolhidos de modo que maximizem a covariância entre o X_k projetado e o alvo projetado, que é \text{Cov}(X_k u_k,Y_k v_k).

        # b) Projete X_k e Y_k nos vetores singulares para obter pontuações: \xi_k = X_k u_k e \omega_k = Y_k v_k

        # c) Regresse X_k em \xi_k, ou seja, encontre um vetor \gamma_k\in \mathbb{R}^d tal que a matriz de posto 1 \xi_k \gamma_k^T seja o mais próximo possível de X_k. Faça o mesmo em Y_k com \omega_k para obter \delta_k. Os vetores \gamma_k e \delta_k são chamados de carregamentos.

        # d) esvazie X_k e Y_k, ou seja, subtraia as aproximações de posto 1: X_{k+1} = X_k - \xi_k \gamma_k^T e Y_{k + 1} = Y_k - \omega_k \delta_k^T. 
        

    # No final, aproximamos X como uma soma de matrizes de posto 1: X = \Xi \Gamma^T onde \Xi \in \mathbb{R}^{n \times K} contém as pontuações em suas colunas, e \Gamma^T \in \mathbb{R}^{K\times d} contém os carregamentos em suas linhas. Da mesma forma para Y, temos Y = \Omega \Delta^T.

    # Observe que as matrizes de pontuação \Xi e \Omega correspondem às projeções dos dados de treinamento X e Y, respectivamente.

    # A etapa a) pode ser realizada de duas maneiras: ou calculando todo o SVD de C e retendo apenas os vetores singulares com os maiores valores singulares, ou calculando diretamente os vetores singulares usando o método da potência (cf seção 11.3 em 1), que corresponde à opção 'nipals' do parâmetro do algoritmo. 



##### 1.8.1.1. Transformando dados

    # Para transformar X em \bar{X}, precisamos encontrar uma matriz de projeção P tal que \bar{X} = XP. Sabemos que para os dados de treinamento, \Xi = XP e X = \Xi \Gamma^T. Definindo P = U(\Gamma^T U)^{-1} onde U é a matriz com o u_k nas colunas, temos XP = X U(\Gamma^T U)^{-1} = \Xi(\Gamma^T U) (\Gamma^T U)^{-1} = \Xi conforme desejado. A matriz de rotação P pode ser acessada a partir do atributo x_rotations_.

    # Da mesma forma, Y pode ser transformado usando a matriz de rotação V(\Delta^T V)^{-1}, acessada através do atributo y_rotations_. 








##### 1.8.1.2. Previsão dos alvos Y 

    # Para prever os alvos de alguns dados X, estamos procurando uma matriz de coeficientes \beta \in R^{d \times t} tal que Y = X\beta.

    # A ideia é tentar prever os alvos transformados \Omega em função das amostras transformadas \Xi, calculando \alpha\in \mathbb{R} tal que \Omega = \alpha \Xi.

    # Então, temos Y = \Omega \Delta^T = \alpha \Xi \Delta^T, e como \Xi são os dados de treinamento transformados, temos que Y = X \alpha P \Delta^T, e como resultado o matriz de coeficiente \beta = \alpha P\Delta^T.

    # \beta pode ser acessado através do atributo coef_. 