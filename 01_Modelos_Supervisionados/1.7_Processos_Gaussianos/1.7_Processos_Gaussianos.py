########## 1.7. Processos Gaussianos ##########




    # VOs Processos Gaussianos (GP) são um método genérico de aprendizado supervisionado projetado para resolver problemas de regressão e classificação probabilística.

    # VAs vantagens dos processos gaussianos são:

        # A previsão interpola as observações (pelo menos para kernels regulares).

        # A predição é probabilística (gaussiana) para que se possa calcular intervalos de confiança empíricos e decidir com base neles se deve-se reajustar (ajuste online, ajuste adaptativo) a previsão em alguma região de interesse.

    # Versátil: diferentes kernels podem ser especificados. Kernels comuns são fornecidos, mas também é possível especificar kernels personalizados.

    # VAs desvantagens dos processos gaussianos incluem:

        # Eles não são esparsos, ou seja, eles usam todas as informações de amostras/características para realizar a previsão.

        # Perdem eficiência em espaços de grande dimensão – nomeadamente quando o número de feições ultrapassa algumas dezenas. 