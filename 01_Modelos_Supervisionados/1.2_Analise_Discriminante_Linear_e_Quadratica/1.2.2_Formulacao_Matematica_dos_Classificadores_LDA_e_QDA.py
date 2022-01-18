########## 1.2.2. Formulação matemática dos classificadores LDA e QDA ##########

    # Tanto LDA quanto QDA podem ser derivados de modelos probabilísticos simples que modelam a distribuição condicional de classe dos dados P(X|y=k) para cada classe k. As previsões podem ser obtidas usando a regra de Bayes, para cada amostra de treinamento x \in \mathcal{R}^d:


        # P(y=k | x) = \frac{P(x | y=k) P(y=k)}{P(x)} = \frac{P(x | y=k) P(y = k )}{ \sum_{l} P(x | y=l) \cdot P(y=l)}


    # e selecionamos a classe k que maximiza essa probabilidade a posteriori.

    # Mais especificamente, para análise discriminante linear e quadrática, P(x|y) é modelado como uma distribuição gaussiana multivariada com densidade:

        # P(x | y=k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}}\exp\left(-\frac{1}{2) } (x-\mu_k)^t \Sigma_k^{-1} (x-\mu_k)\right)

    # onde d é o número de características. 




##### 1.2.2.1. QDA

    # De acordo com o modelo acima, o logaritmo da posterior é:

        # \begin{split}\log P(y=k | x) &= \log P(x | y=k) + \log P(y = k) + Cst \\
        # &= -\frac{1}{2} \log |\Sigma_k| -\frac{1}{2} (x-\mu_k)^t \Sigma_k^{-1} (x-\mu_k) + \log P(y = k) + Cst,\end{split}


    # onde o termo constante Cst corresponde ao denominador P(x), além de outros termos constantes da Gaussiana. A classe prevista é aquela que maximiza esse log-posterior.

    # Nota: Relação com Gaussian Naive Bayes
    # Se no modelo QDA se assume que as matrizes de covariância são diagonais, então as entradas são consideradas condicionalmente independentes em cada classe, e o classificador resultante é equivalente ao classificador Gaussiano Naive Bayes naive_bayes.GaussianNB. 








##### 1.2.2.2. LDA 

    # LDA é um caso especial de QDA, onde assume-se que as Gaussianas para cada classe compartilham a mesma matriz de covariância: \Sigma_k = \Sigma para todos os k. Isso reduz o log posterior a:

        # \log P(y=k | x) = -\frac{1}{2} (x-\mu_k)^t \Sigma^{-1} (x-\mu_k) + \log P(y = k) + Cst.

    # O termo (x-\mu_k)^t \Sigma^{-1} (x-\mu_k) corresponde à Distância de Mahalanobis entre a amostra x e a média \mu_k. A distância de Mahalanobis informa o quão próximo x está de \mu_k, enquanto também contabiliza a variância de cada característica. Podemos, assim, interpretar LDA como a atribuição de x à classe cuja média é a mais próxima em termos da distância de Mahalanobis, ao mesmo tempo em que leva em conta as probabilidades anteriores da classe.

    # O log-posterior de LDA também pode ser escrito 3 como:

        # \log P(y=k | x) = \omega_k^t x + \omega_{k0} + Cst.

    # onde \omega_k = \Sigma^{-1} \mu_k e \omega_{k0} =-\frac{1}{2} \mu_k^t\Sigma^{-1}\mu_k + \log P (y = k). Essas quantidades correspondem aos atributos coef_ e intercept_, respectivamente.

    # A partir da fórmula acima, fica claro que o LDA tem uma superfície de decisão linear. No caso de QDA, não há suposições sobre as matrizes de covariância \Sigma_k das Gaussianas, levando a superfícies de decisão quadráticas. Veja 1 para mais detalhes. 