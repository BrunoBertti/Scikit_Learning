########## 2.1.1. Mistura Gaussiana  ##########

    # O objeto GaussianMixture implementa o algoritmo de maximização de expectativa (EM) para ajustar modelos de mistura de gaussianos. Ele também pode desenhar elipsóides de confiança para modelos multivariados e calcular o Critério de Informação Bayesiano para avaliar o número de clusters nos dados. É fornecido um método GaussianMixture.fit que aprende um modelo de mistura gaussiana a partir dos dados do treino. Dados de teste dados, ele pode atribuir a cada amostra o Gaussiano ao qual provavelmente pertence, usando o método GaussianMixture.predict.

    # O GaussianMixture vem com diferentes opções para restringir a covariância das classes de diferença estimadas: covariância esférica, diagonal, ligada ou total. 


        # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
    
    

    ## Exemplos:

    ## See GMM covariances for an example of using the Gaussian mixture as clustering on the iris dataset. (https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py)

    ## See Density Estimation for a Gaussian mixture for an example on plotting the density estimation. (https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py)


##### 2.1.1.1. Prós e contras da classe GaussianMixture 

##### 2.1.1.1.1. Pros

    # Velocidade: É o algoritmo mais rápido para aprender modelos de mistura

    # Agnóstico1: Como esse algoritmo maximiza apenas a probabilidade, ele não enviesará as médias para zero, nem enviesará os tamanhos dos agrupamentos para que tenham estruturas específicas que podem ou não se aplicar. 


##### 2.1.1.1.2. Contras

    # Singularidades: Quando se tem um número insuficiente de pontos por mistura, estimar as matrizes de covariância torna-se difícil e o algoritmo diverge e encontra soluções com probabilidade infinita, a menos que regularize as covariâncias artificialmente.

    # Número de componentes: Este algoritmo sempre usará todos os componentes aos quais tem acesso, necessitando de critérios teóricos de dados ou informações para decidir quantos componentes usar na ausência de pistas externas. 

##### 2.1.1.2. Seleção do número de componentes em um modelo clássico de mistura gaussiana 

    # O critério BIC pode ser usado para selecionar o número de componentes em uma Mistura Gaussiana de maneira eficiente. Em teoria, ele recupera o número verdadeiro de componentes apenas no regime assintótico (ou seja, se muitos dados estiverem disponíveis e assumindo que os dados foram realmente gerados i.i.d. a partir de uma mistura de distribuição gaussiana). Observe que o uso de uma mistura Gaussiana Bayesiana Variacional evita a especificação do número de componentes para um modelo de mistura Gaussiana. 

        # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html


    ## Exemplos:

    ## See Gaussian Mixture Model Selection for an example of model selection performed with classical Gaussian mixture. (https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py)

##### 2.1.1.3. Algoritmo de estimativa - maximização da expectativa 

    # A principal dificuldade em aprender modelos de mistura gaussiana a partir de dados não rotulados é que geralmente não se sabe quais pontos vieram de qual componente latente (se alguém tiver acesso a esta informação, fica muito fácil ajustar uma distribuição gaussiana separada para cada conjunto de pontos). A maximização da expectativa é um algoritmo estatístico bem fundamentado para contornar esse problema por um processo iterativo. O primeiro assume componentes aleatórios (centrados aleatoriamente em pontos de dados, aprendidos de k-means, ou mesmo apenas normalmente distribuídos em torno da origem) e calcula para cada ponto uma probabilidade de ser gerado por cada componente do modelo. Em seguida, ajusta-se os parâmetros para maximizar a probabilidade dos dados dados essas atribuições. A repetição desse processo sempre convergirá para um ótimo local. 