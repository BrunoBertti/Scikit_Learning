########## 1.4.7 Formulação Matemática ##########

    # Uma máquina de vetores de suporte constrói um hiperplano ou conjunto de hiperplanos em um espaço dimensional alto ou infinito, que pode ser usado para classificação, regressão ou outras tarefas. Intuitivamente, uma boa separação é obtida pelo hiperplano que possui a maior distância aos pontos de dados de treinamento mais próximos de qualquer classe (a chamada margem funcional), pois em geral quanto maior a margem menor o erro de generalização do classificador. A figura abaixo mostra a função de decisão para um problema separável linearmente, com três amostras nos limites da margem, chamados de “vetores de suporte”:

        # https://scikit-learn.org/stable/_images/sphx_glr_plot_separating_hyperplane_001.png

    # Em geral, quando o problema não é linearmente separável, os vetores de suporte são as amostras dentro dos limites da margem.

    # Recomendamos 13 e 14 como boas referências para a teoria e os aspectos práticos dos SVMs. 


##### 1.4.7.1 SVC
    
    # Dados os vetores de treinamento x_i \ in \ mathbb {R} ^ p, i = 1,…, n, em duas classes, e um vetor y \ in \ {1, -1 \} ^ n, nosso objetivo é encontrar w \ in \ mathbb {R} ^ p e b \ in \ mathbb {R} tal que a predição dada por \ text {sinal} (w ^ T \ phi (x) + b) é correta para a maioria das amostras. 

    # O SVC resolve o seguinte problema primário: 

        # \ begin {align} \ begin {alinhados} \ min_ {w, b, \ zeta} \ frac {1} {2} w ^ T w + C \ sum_ {i = 1} ^ {n} \ zeta_i \\\ begin {split} \ textrm {subject to} & y_i (w ^ T \ phi (x_i) + b) \ geq 1 - \ zeta_i, \\ & \ zeta_i \ geq 0, i = 1, ..., n \ end {split} \ end {alinhado} \ end {align} 

    # Intuitivamente, estamos tentando maximizar a margem (minimizando || w || ^ 2 = w ^ Tw), enquanto incorremos em uma penalidade quando uma amostra é classificada incorretamente ou dentro do limite da margem. Idealmente, o valor y_i (w ^ T \ phi (x_i) + b) seria> 1 para todas as amostras, o que indica uma previsão perfeita. Mas geralmente os problemas nem sempre são perfeitamente separáveis com um hiperplano, então permitimos que algumas amostras estejam a uma distância \ zeta_i de seu limite de margem correto. O termo de penalidade C controla a força dessa penalidade e, como resultado, atua como um parâmetro de regularização inverso (consulte a nota abaixo). 

    # O problema duplo para o primal é 

        # \begin{align}\begin{aligned}\min_{\alpha} \frac{1}{2} \alpha^T Q \alpha - e^T \alpha\\\begin{split} \textrm {subject to } & y^T \alpha = 0\\& 0 \leq \alpha_i \leq C, i=1, ..., n\end{split}\end{aligned}\end{align}

    # onde e é o vetor de todos os uns, e Q é uma matriz semidefinida positiva n por n, Q_ {ij} \ equiv y_i y_j K (x_i, x_j), onde K (x_i, x_j) = \ phi (x_i) ^ T \ phi (x_j) é o kernel. Os termos \ alpha_i são chamados de coeficientes duais e são limitados por C no topo. Essa representação dual destaca o fato de que os vetores de treinamento são implicitamente mapeados em um espaço dimensional superior (talvez infinito) pela função \ phi: veja o truque do kernel.

    # Uma vez que o problema de otimização é resolvido, a saída de função_de_ decisão para uma determinada amostra x torna-se: 

        # \sum_{i\in SV} y_i \alpha_i K(x_i, x) + b,

    # e a classe prevista corresponde ao seu signo. Precisamos apenas somar os vetores de suporte (ou seja, as amostras que estão dentro da margem) porque os coeficientes duais \ alpha_i são zero para as outras amostras.

    # Esses parâmetros podem ser acessados por meio dos atributos dual_coef_ que contém o produto y_i \ alpha_i, support_vectors_ que contém os vetores de suporte e intercept_ que contém o termo independente b

    ## OBS: Nota Enquanto os modelos SVM derivados de libsvm e liblinear usam C como parâmetro de regularização, a maioria dos outros estimadores usa alfa. A equivalência exata entre a quantidade de regularização de dois modelos depende da função objetivo exata otimizada pelo modelo. Por exemplo, quando o estimador usado é a regressão de Ridge, a relação entre eles é dada como C = \ frac {1} {alfa}

##### 1.4.7.2 LinearSVC

    # O problema primordial pode ser formulado de forma equivalente como

        # \min_ {w, b} \frac{1}{2} w^T w + C \sum_{i=1}\max(0, 1 - y_i (w^T \phi(x_i) + b)),

    # onde fazemos uso da perda de dobradiça. Esta é a forma que é diretamente otimizada pelo LinearSVC, mas ao contrário da forma dual, esta não envolve produtos internos entre as amostras, então o famoso truque do kernel não pode ser aplicado. É por isso que apenas o kernel linear é compatível com LinearSVC (\ phi é a função de identidade)

##### 1.4.7.3 NuSVC

    # A formulação \ nu-SVC 15 é uma reparametrização do C-SVC e, portanto, matematicamente equivalente.


    # Introduzimos um novo parâmetro \ nu (em vez de C) que controla o número de vetores de suporte e erros de margem: \ nu \ in (0, 1] é um limite superior na fração de erros de margem e um limite inferior da fração de vetores de suporte Um erro de margem corresponde a uma amostra que está do lado errado de seu limite de margem: ela está mal classificada ou está classificada corretamente, mas não está além da margem.

##### 1.4.7.4 SVR

    # Dados os vetores de treinamento x_i \ in \ mathbb {R} ^ p, i = 1,…, n, e um vetor y \ in \ mathbb {R} ^ n \ varejpsilon-SVR resolve o seguinte problema primordial:

        # \begin{align}\begin{aligned}\min_ {w, b, \zeta, \zeta^*} \frac{1}{2} w^T w + C \sum_{i=1}^{n} (\zeta_i + \zeta_i^*)\\\begin{split}\textrm {subject to } & y_i - w^T \phi (x_i) - b \leq \varepsilon + \zeta_i,\\& w^T \phi (x_i) + b - y_i \leq \varepsilon + \zeta_i^*,\\& \zeta_i, \zeta_i^* \geq 0, i=1, ..., n\end{split}\end{aligned}\end{align}

    # Aqui, estamos penalizando amostras cuja previsão está pelo menos longe de seu verdadeiro alvo. Essas amostras penalizam o objetivo por \ zeta_i ou \ zeta_i ^ *, dependendo se suas previsões estão acima ou abaixo do tubo \ varepsilon.

    # O problema duplo é

        # \begin{align}\begin{aligned}\min_{\alpha, \alpha^*} \frac{1}{2} (\alpha - \alpha^*)^T Q (\alpha - \alpha^*) + \varepsilon e^T (\alpha + \alpha^*) - y^T (\alpha - \alpha^*)\\\begin{split}\textrm {subject to } & e^T (\alpha - \alpha^*) = 0\\& 0 \leq \alpha_i, \alpha_i^* \leq C, i=1, ..., n\end{split}\end{aligned}\end{align}

    # onde e é o vetor de todos os uns, Q é uma matriz semidefinida positiva, Q_ {ij} \ equiv K (x_i, x_j) = \ phi (x_i) ^ T \ phi (x_j) é o kernel. Aqui, os vetores de treinamento são implicitamente mapeados em um espaço dimensional superior (talvez infinito) pela função \ phi. 

    # A previsão é: 

        # \sum_{i \in SV}(\alpha_i - \alpha_i^*) K(x_i, x) + b

    # Esses parâmetros podem ser acessados por meio dos atributos dual_coef_ que contém a diferença \ alpha_i - \ alpha_i ^ *, support_vectors_ que contém os vetores de suporte e intercept_ que contém o termo independente b 


##### 1.4.7.5 LinearSVR

    # O problema primordial pode ser formulado de forma equivalente como 

        # \min_ {w, b} \frac{1}{2} w^T w + C \sum_{i=1}\max(0, |y_i - (w^T \phi(x_i) + b)| - \varepsilon),
    
    # onde fazemos uso da perda insensível a épsilon, ou seja, erros menores que e são ignorados. Este é o formulário que é otimizado diretamente pelo LinearSVR. 

