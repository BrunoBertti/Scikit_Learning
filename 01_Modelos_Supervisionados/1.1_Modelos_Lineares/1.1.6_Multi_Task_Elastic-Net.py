########## 1.1.6 Multi Task Elastic-Net ##########

    # O MultiTaskElasticNet é um modelo elastic-net que estima os coeficientes esparsos por multiplos problemas de regressão conjuntamente: Y é uma matrzi 2d de fomra (2_eamples, n_tasks). A restrição é que as variáveis selecionadas são os mesmo para todos os problemas de regressão. também chamados de tarefas.

    # Matematicamente, consiste em um modelo linear treinado com l1l2 norm e um norm l2 misturados para a regularização. A função objetivo é minimizar:

        # min 1/2nsamples ||WX - Y||2Fro + alpha p||W||21 + alpha(1-p)/2 ||W||2Fro
    
    # A implementação das classes MultiTaskElasticNet usa coordenadas decendentes como o algoritmo para treinar os coeficientes.

    # A classe MultiTaskElasticNetCV pode ser usada para definir os parâmetros alpha () e l1_ratio () por validação cruzada. 