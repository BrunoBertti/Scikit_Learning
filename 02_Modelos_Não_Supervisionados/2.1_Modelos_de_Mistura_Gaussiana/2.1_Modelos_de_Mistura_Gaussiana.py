########## 2.1. Modelos de mistura gaussiana  ##########

    # sklearn.mixture é um pacote que permite aprender Modelos de Mistura Gaussiana (matrizes diagonais, esféricas, amarradas e de covariância total suportadas), amostrá-los e estimá-los a partir dos dados. Recursos para ajudar a determinar o número apropriado de componentes também são fornecidos. 

        # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html

    # Um modelo de mistura gaussiana é um modelo probabilístico que assume que todos os pontos de dados são gerados a partir de uma mistura de um número finito de distribuições gaussianas com parâmetros desconhecidos. Pode-se pensar em modelos de mistura como generalizando agrupamento de k-médias para incorporar informações sobre a estrutura de covariância dos dados, bem como os centros das gaussianas latentes.

    # O Scikit-learn implementa diferentes classes para estimar modelos de mistura gaussiana, que correspondem a diferentes estratégias de estimativa, detalhadas a seguir. 