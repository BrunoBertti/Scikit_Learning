########## 1.7.5. Kernels para processos gaussianos  ##########

    # Kernels (também chamados de “funções de covariância” no contexto de GPs) são um ingrediente crucial de GPs que determinam a forma de anterior e posterior do GP. Eles codificam as suposições sobre a função que está sendo aprendida, definindo a “semelhança” de dois pontos de dados combinados com a suposição de que pontos de dados semelhantes devem ter valores de destino semelhantes. Duas categorias de kernels podem ser distinguidas: kernels estacionários dependem apenas da distância de dois pontos de dados e não de seus valores absolutos k(x_i, x_j)= k(d(x_i, x_j)) e, portanto, são invariantes para traduções no espaço de entrada , enquanto os kernels não estacionários dependem também dos valores específicos dos pontos de dados. Kernels estacionários podem ainda ser subdivididos em kernels isotrópicos e anisotrópicos, onde kernels isotrópicos também são invariantes a rotações no espaço de entrada. Para mais detalhes, consulte o Capítulo 4 de [RW2006]. Para obter orientação sobre como combinar melhor os diferentes kernels, consulte [Duv2014]. 


##### 1.7.5.1. API do kernel de processo gaussiano

    # O principal uso de um Kernel é calcular a covariância do GP entre os pontos de dados. Para isso, o método __call__ do kernel pode ser chamado. Este método pode ser usado para calcular a “auto-covariância” de todos os pares de pontos de dados em uma matriz 2d X ou a “covariância cruzada” de todas as combinações de pontos de dados de uma matriz 2d X com pontos de dados em uma matriz 2d Y. A seguinte identidade é válida para todos os kernels k (exceto para o WhiteKernel): k(X) == K(X, Y=X)

    # Se apenas a diagonal da autocovariância estiver sendo usada, o método diag() de um kernel pode ser chamado, que é mais eficiente computacionalmente do que a chamada equivalente a __call__: np.diag(k(X, X)) == k.diag(X)

    # Kernels são parametrizados por um vetor \theta de hiperparâmetros. Esses hiperparâmetros podem, por exemplo, controlar escalas de comprimento ou periodicidade de um kernel (veja abaixo). Todos os kernels suportam gradientes analíticos de computação da auto-covariância do kernel em relação a log(\theta) por meio da configuração eval_gradient=True no método __call__. Ou seja, um array (len(X), len(X), len(theta)) é retornado onde a entrada [i, j, l] contém \frac{\partial k_\theta(x_i, x_j)}{\ log parcial(\theta_l)}. Este gradiente é usado pelo processo gaussiano (tanto regressor quanto classificador) no cálculo do gradiente de log-marginal-likelihood, que por sua vez é usado para determinar o valor de \theta, que maximiza o log-marginal-likelihood, via gradiente subida. Para cada hiperparâmetro, o valor inicial e os limites precisam ser especificados ao criar uma instância do kernel. O valor atual de \theta pode ser obtido e definido através da propriedade theta do objeto do kernel. Além disso, os limites dos hiperparâmetros podem ser acessados ​​pelos limites de propriedade do kernel. Observe que ambas as propriedades (teta e limites) retornam valores transformados em log dos valores usados ​​internamente, uma vez que normalmente são mais passíveis de otimização baseada em gradiente. A especificação de cada hiperparâmetro é armazenada na forma de uma instância de Hiperparâmetro no respectivo kernel. Observe que um kernel usando um hiperparâmetro com o nome “x” deve ter os atributos self.x e self.x_bounds.

    # A classe base abstrata para todos os kernels é Kernel. Kernel implementa uma interface semelhante ao Estimator, fornecendo os métodos get_params(), set_params() e clone(). Isso permite definir valores de kernel também por meio de meta-estimadores, como Pipeline ou GridSearch. Observe que devido à estrutura aninhada dos kernels (aplicando os operadores do kernel, veja abaixo), os nomes dos parâmetros do kernel podem se tornar relativamente complicados. Em geral, para um operador binário do kernel, os parâmetros do operando esquerdo são prefixados com k1__ e os parâmetros do operando direito com k2__. Um método de conveniência adicional é clone_with_theta(theta), que retorna uma versão clonada do kernel, mas com os hiperparâmetros definidos como theta. Um exemplo ilustrativo: 

from sklearn.gaussian_process.kernels import ConstantKernel, RBF
kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
for hyperparameter in kernel.hyperparameters: print(hyperparameter)
params = kernel.get_params()
for key in sorted(params): print("%s : %s" % (key, params[key]))
print(kernel.theta)  # Nota: log transformado 
print(kernel.bounds)  # Nota: log transformado 


    # Todos os kernels de processo gaussianos são interoperáveis com sklearn.metrics.pairwise e vice-versa: instâncias de subclasses de Kernel podem ser passadas como métrica para pairwise_kernels de sklearn.metrics.pairwise. Além disso, as funções do kernel do pairwise podem ser usadas como kernels GP usando a classe wrapper PairwiseKernel. A única ressalva é que o gradiente dos hiperparâmetros não é analítico, mas numérico e todos esses kernels suportam apenas distâncias isotrópicas. O parâmetro gama é considerado um hiperparâmetro e pode ser otimizado. Os outros parâmetros do kernel são definidos diretamente na inicialização e são mantidos fixos. 








##### 1.7.5.2. Kernels básicos


    # O kernel ConstantKernel pode ser usado como parte de um kernel Product onde dimensiona a magnitude do outro fator (kernel) ou como parte de um kernel Sum, onde modifica a média do processo Gaussiano. Depende de um parâmetro constant\_value. É definido como:

        # k(x_i, x_j) = constante\_valor \;\forall\; x_1, x_2

    # O principal caso de uso do kernel WhiteKernel é como parte de um kernel de soma onde explica o componente de ruído do sinal. Ajustar seu parâmetro noise\_level corresponde a estimar o nível de ruído. É definido como:

        # k(x_i, x_j) = ruído\_nível \text{ if } x_i == x_j \text{ else } 0 







##### 1.7.5.3. Operadores de kernel

    # Os operadores de kernel pegam um ou dois kernels básicos e os combinam em um novo kernel. O kernel Sum pega dois kernels k_1 e k_2 e os combina via k_{sum}(X, Y) = k_1(X, Y) + k_2(X, Y). O kernel Product pega dois kernels k_1 e k_2 e os combina via k_{product}(X, Y) = k_1(X, Y) * k_2(X, Y). O kernel Exponenciation recebe um kernel base e um parâmetro escalar p e os combina via k_{exp}(X, Y) = k(X, Y)^p. Observe que os métodos mágicos __add__, __mul___ e __pow__ são sobrescritos nos objetos Kernel, então pode-se usar, por exemplo, RBF() + RBF() como atalho para Sum(RBF(), RBF()). 






##### 1.7.5.4. Kernel de função de base radial (RBF)

    # O kernel RBF é um kernel estacionário. Também é conhecido como kernel “exponencial ao quadrado”. Ele é parametrizado por um parâmetro de escala de comprimento l>0, que pode ser um escalar (variante isotrópica do kernel) ou um vetor com o mesmo número de dimensões das entradas x (variante anisotrópica do kernel). O núcleo é dado por: 

        # k(x_i, x_j) = \text{exp}\left(- \frac{d(x_i, x_j)^2}{2l^2} \right)

    # onde d(\cdot, \cdot) é a distância euclidiana. Esse kernel é infinitamente diferenciável, o que implica que GPs com esse kernel como função de covariância têm derivadas quadradas médias de todas as ordens e, portanto, são muito suaves. O anterior e o posterior de um GP resultante de um kernel RBF são mostrados na figura a seguir: 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html

        





##### 1.7.5.5. Matérn kernel


    # O kernel Matern é um kernel estacionário e uma generalização do kernel RBF. Tem um parâmetro adicional \nu que controla a suavidade da função resultante. Ele é parametrizado por um parâmetro de escala de comprimento l>0, que pode ser um escalar (variante isotrópica do kernel) ou um vetor com o mesmo número de dimensões das entradas x (variante anisotrópica do kernel). O núcleo é dado por: 

        # k(x_i, x_j) = \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(\frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg)^\nu K_\nu\Bigg(\frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg),

    # onde d(\cdot,\cdot) é a distância euclidiana, K_\nu(\cdot) é uma função de Bessel modificada e \Gamma(\cdot) é a função gama. Como \nu\rightarrow\infty, o kernel Matérn converge para o kernel RBF. Quando \nu = 1/2, o kernel Matérn torna-se idêntico ao kernel exponencial absoluto, ou seja,

        # k(x_i, x_j) = \exp \Bigg(- \frac{1}{l} d(x_i, x_j) \Bigg) \quad \quad \nu= \tfrac{1}{2}

    # Em particular, \nu = 3/2:

        # k(x_i, x_j) = \Bigg(1 + \frac{\sqrt{3}}{l} d(x_i, x_j)\Bigg) \exp \Bigg(-\frac{\sqrt{3}}{l } d(x_i , x_j ) \ Bigg) \quad \quad \nu= \tfrac{3}{2}

    # e \nu = 5/2:

        # k(x_i, x_j) = \ Bigg(1 + \frac{\sqrt{5}}{l} d(x_i , x_j ) +\frac{5}{3l} d(x_i , x_j )^2 \Bigg) \exp \Bigg(-\frac{\sqrt{5}}{l} d(x_i , x_j ) \Bigg) \quad \quad \nu= \tfrac{5}{2}


    # são escolhas populares para aprender funções que não são infinitamente diferenciáveis ​​(como assumido pelo kernel RBF), mas pelo menos uma vez (\nu =3/2) ou duas vezes diferenciável (\nu = 5/2).

    # A flexibilidade de controlar a suavidade da função aprendida permite a adaptação às propriedades da verdadeira relação funcional subjacente. O anterior e o posterior de um GP resultante de um kernel Matérn são mostrados na figura a seguir: 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html

    # Veja [RW2006], pp84 para mais detalhes sobre as diferentes variantes do kernel Matérn. 



##### 1.7.5.6. Núcleo quadrático racional


    # O kernel RationalQuadratic pode ser visto como uma mistura de escala (uma soma infinita) de kernels RBF com diferentes escalas de comprimento características. Ele é parametrizado por um parâmetro de escala de comprimento l>0 e um parâmetro de mistura de escala \alpha>0 Somente a variante isotrópica onde l é um escalar é suportada no momento. O núcleo é dado por:


        # k(x_i, x_j) = \left(1 + \frac{d(x_i, x_j)^2}{2\alpha l^2}\right)^{-\alpha}


    # O anterior e o posterior de um GP resultante de um kernel RationalQuadratic são mostrados na figura a seguir: 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html




##### 1.7.5.7. Kernel Exp-Sine-Squared



    # O kernel ExpSineSquared permite modelar funções periódicas. É parametrizado por um parâmetro de escala de comprimento l>0 e um parâmetro de periodicidade p>0. Apenas a variante isotrópica onde l é um escalar é suportada no momento. O núcleo é dado por:


        # k(x_i, x_j) = \text{exp}\left(- \frac{ 2\sin^2(\pi d(x_i, x_j) / p) }{ l^ 2} \right)

    # O anterior e o posterior de um GP resultante de um kernel ExpSineSquared são mostrados na figura a seguir: 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html


##### 1.7.5.8. Kernel de produto ponto

    # O kernel DotProduct não é estacionário e pode ser obtido a partir de regressão linear colocando N(0, 1) a priori nos coeficientes de x_d (d = 1, . . . , D) e a a priori de N(0, \sigma_0^ 2) no viés. O kernel DotProduct é invariável a uma rotação das coordenadas em torno da origem, mas não às traduções. É parametrizado por um parâmetro \sigma_0^2. Para \sigma_0^2 = 0, o kernel é chamado de kernel linear homogêneo, caso contrário, não é homogêneo. O núcleo é dado por

        # k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

    # O kernel DotProduct é comumente combinado com exponenciação. Um exemplo com expoente 2 é mostrado na figura a seguir:

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html 




##### 1.7.5.9. Referências 



    ## Referências:

    ## RW2006(1,2,3,4,5,6) Carl Eduard Rasmussen and Christopher K.I. Williams, “Gaussian Processes for Machine Learning”, MIT Press 2006, Link to an official complete PDF version of the book here . (http://www.gaussianprocess.org/gpml/chapters/RW.pdf)

    ## Duv2014 David Duvenaud, “The Kernel Cookbook: Advice on Covariance functions”, 2014, Link . (https://www.cs.toronto.edu/~duvenaud/cookbook/)