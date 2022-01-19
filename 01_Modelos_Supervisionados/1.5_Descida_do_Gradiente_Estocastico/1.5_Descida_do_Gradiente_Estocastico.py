########## 1.5. Descida do Gradiente Estocástico ##########

    # Stochastic Gradient Descent (SGD) é uma abordagem simples, mas muito eficiente, para ajustar classificadores lineares e regressores sob funções de perda convexa, como máquinas de vetor de suporte (linear) e regressão logística. Embora o SGD esteja presente na comunidade de aprendizado de máquina há muito tempo, ele recebeu uma atenção considerável recentemente no contexto do aprendizado em larga escala.

    # O SGD foi aplicado com sucesso a problemas de aprendizado de máquina esparsos e em larga escala, frequentemente encontrados na classificação de texto e no processamento de linguagem natural. Dado que os dados são escassos, os classificadores neste módulo são facilmente dimensionados para problemas com mais de 10^5 exemplos de treinamento e mais de 10^5 recursos.

    # A rigor, SGD é apenas uma técnica de otimização e não corresponde a uma família específica de modelos de aprendizado de máquina. É apenas uma maneira de treinar um modelo. Muitas vezes, uma instância de SGDClassifier ou SGDRegressor terá um estimador equivalente na API scikit-learn, potencialmente usando uma técnica de otimização diferente. Por exemplo, usar SGDClassifier(loss='log') resulta em regressão logística, ou seja, um modelo equivalente a LogisticRegression que é ajustado via SGD em vez de ser ajustado por um dos outros solucionadores em LogisticRegression. Da mesma forma, SGDRegressor(loss='squared_error', penalty='l2') e Ridge resolvem o mesmo problema de otimização, por meios diferentes.

    # As vantagens da descida de gradiente estocástico são: 


        # Eficiência.

        # Facilidade de implementação (muitas oportunidades para ajuste de código).


    # As desvantagens da descida do gradiente estocástico incluem:

        # O SGD requer vários hiperparâmetros, como o parâmetro de regularização e o número de iterações.

        # O SGD é sensível ao dimensionamento de recursos. 


    # Aviso: Certifique-se de permutar (embaralhar) seus dados de treinamento antes de ajustar o modelo ou usar shuffle=True para embaralhar após cada iteração (usado por padrão). Além disso, idealmente, os recursos devem ser padronizados usando, por exemplo, make_pipeline(StandardScaler(), SGDClassifier()) (veja Pipelines). 

