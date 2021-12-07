########## 1.1.12 Regressão Linear Generalizada  ##########

    # Os Modelos Lineares Generalizados (GLM) estendem os modelos lineares de duas maneiras 10. Primeiro, os valores previstos y ^ estão ligados a uma combinação linear das variáveis de entrada X por meio de uma função de ligação inversa como h.

        # y^ (w,X) = h(Xw)

    # Em segundo lugar, a função de perda quadrada é substituída pelo desvio da unidade d de uma distribuição na família exponencial (ou mais precisamente, um modelo de dispersão exponencial reprodutiva (EDM) 11). 

    # O problema de minimização torna-se: 

        # min 1/2nsamples somatório d(yi, y^ i) + alpha/2||w||2

    # onde alfa é a penalidade de regularização L2. Quando os pesos da amostra são fornecidos, a média se torna uma média ponderada.

    # A tabela a seguir lista alguns EDMs específicos e seus desvios de unidade (todos são instâncias da família Tweedie): 

    # Distribuição                  Dominio do alvo             Desvio de Unidade d(y, y^)
    # Normal                        y pertence (-inf, infito)   (y-y^)^2
    # Poisson                       y pertence (0, infito)      2(y log y/y^ - y + y^)
    # Gamma                         y pertence (0, infito)      2(logy^/y + y/y^ -1)
    # Gauisiana inversa             y pertence (0, infito)      (y-y^)^2 / y*y^^2

    # As funções de densidade de probabilidade (PDF) dessas distribuições são ilustradas na figura a seguir, 

    # PDF de uma variável aleatória Y seguindo distribuições de Poisson, Tweedie (potência = 1,5) e Gama com diferentes valores médios (). Observe a massa do ponto em para a distribuição de Poisson e a distribuição de Tweedie (potência = 1,5), mas não para a distribuição Gama, que tem um domínio alvo estritamente positivo. 


    # A escolha da distribuição depende do problema em questão: 

    # Se os valores de destino forem contagens (valor inteiro não negativo) ou frequências relativas (não negativo), você pode usar um desvio de Poisson com log-link.

    # Se os valores de destino forem avaliados positivamente e distorcidos, você pode tentar um desvio gama com log-link.

    # Se os valores alvo parecem ter cauda mais pesada do que uma distribuição Gama, você pode tentar um desvio Gaussiano Inverso (ou poderes de variância ainda mais altos da família Tweedie). 


    # Exemplos de casos de uso incluem: 

    # Modelagem agrícola / meteorológica: número de eventos de chuva por ano (Poisson), quantidade de chuva por evento (Gamma), precipitação total por ano (Tweedie / Compound Poisson Gamma).

    # Modelagem de risco / precificação de apólice de seguro: número de eventos de sinistro / segurado por ano (Poisson), custo por evento (Gamma), custo total por segurado por ano (Tweedie / Compound Poisson Gamma).

    # Manutenção preditiva: número de eventos de interrupção de produção por ano (Poisson), duração da interrupção (Gamma), tempo total de interrupção por ano (Tweedie / Composto Poisson Gamma). 

    ## Referências:

    ## McCullagh, Peter; Nelder, John (1989). Generalized Linear Models, Second Edition. Boca Raton: Chapman and Hall/CRC. ISBN 0-412-31760-5.

    ## Jørgensen, B. (1992). The theory of exponential dispersion models and analysis of deviance. Monografias de matemática, no. 51. See also Exponential dispersion model. (https://en.wikipedia.org/wiki/Exponential_dispersion_model)



##### 1.1.12.1  Uso 

    # power = 0 : Distribuição normal. Estimadores específicos como Ridge, ElasticNet são geralmente mais apropriados neste caso.

    # power = 1 : Distribuição de Poisson. PoissonRegressor é exposto por conveniência. No entanto, é estritamente equivalente a TweedieRegressor (potência = 1, link = 'log').

    # power = 2 : Distribuição gama. GammaRegressor é exposto por conveniência. No entanto, é estritamente equivalente a TweedieRegressor (power = 2, link = 'log').

    # power = 3 : Distribuição inversa de Gauss. 

    # A função de link é determinada pelo parâmetro de link. 

    # Exemplo:

from sklearn.linear_model import TweedieRegressor
reg = TweedieRegressor(power=1, alpha=0.5, link='log')
print(reg.fit([[0, 0], [0, 1], [2, 2]], [0, 1, 2]))

print(reg.coef_)

print(reg.intercept_)


    ## Exemplos:

     ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html#sphx-glr-auto-examples-linear-model-plot-poisson-regression-non-normal-loss-py

     ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html#sphx-glr-auto-examples-linear-model-plot-tweedie-regression-insurance-claims-py



##### 1.1.12.2  Considerações Práticas

    # A matriz de recursos X deve ser padronizada antes do ajuste. Isso garante que a penalidade trate os recursos igualmente.

    # Uma vez que o preditor linear Xw pode ser negativo e as distribuições de Poisson, Gama e Gaussiana Inversa não suportam valores negativos, é necessário aplicar uma função de ligação inversa que garanta a não negatividade. Por exemplo, com link = 'log', a função de link inverso torna-se h (Xw) = exp (Xw).

    # Se você deseja modelar uma frequência relativa, ou seja, contagens por exposição (tempo, volume, ...), você pode fazer isso usando uma distribuição de Poisson e passando y = contagens / valores alvo de exposição junto com os pesos de amostra de exposição. Para obter um exemplo concreto, consulte, por exemplo, Regressão Tweedie em sinistros de seguros.

    # Ao realizar a validação cruzada para o parâmetro de potência de TweedieRegressor, é aconselhável especificar uma função de pontuação explícita, porque o marcador padrão TweedieRegressor.score é uma função de potência em si. 