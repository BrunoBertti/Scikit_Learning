########## 2.7.2. Detecção de Novidades ##########


    # Considere um conjunto de dados de n observações da mesma distribuição descrita por p características. Considere agora que adicionamos mais uma observação a esse conjunto de dados. A nova observação é tão diferente das outras que podemos duvidar que seja regular? (ou seja, vem da mesma distribuição?) Ou, pelo contrário, é tão semelhante ao outro que não podemos distingui-lo das observações originais? Esta é a questão abordada pelas ferramentas e métodos de detecção de novidades.

    # Em geral, trata-se de aprender uma fronteira aproximada e grosseira que delimita o contorno da distribuição das observações iniciais, plotada no espaço p-dimensional embutido. Então, se outras observações estiverem dentro do subespaço delimitado por fronteira, elas são consideradas como provenientes da mesma população que as observações iniciais. Caso contrário, se estiverem fora da fronteira, podemos dizer que são anormais com certa confiança em nossa avaliação.

    # O SVM One-Class foi introduzido por Schölkopf et al. para esse fim e implementado no módulo Support Vector Machines no objeto svm.OneClassSVM. Requer a escolha de um kernel e um parâmetro escalar para definir uma fronteira. O kernel RBF é geralmente escolhido, embora não exista nenhuma fórmula ou algoritmo exato para definir seu parâmetro de largura de banda. Este é o padrão na implementação do scikit-learn. O parâmetro nu, também conhecido como margem do SVM One-Class, corresponde à probabilidade de encontrar uma nova, mas regular, observação fora da fronteira. 


    ## Referências:

    ## Estimating the support of a high-dimensional distribution Schölkopf, Bernhard, et al. Neural computation 13.7 (2001): 1443-1471. (http://www.recognition.mccme.ru/pub/papers/SVM/sch99estimating.pdf)


    ## Exemplos:

    ## See One-class SVM with non-linear kernel (RBF) for visualizing the frontier learned around some data by a svm.OneClassSVM object. (See One-class SVM with non-linear kernel (RBF) for visualizing the frontier learned around some data by a svm.OneClassSVM object.)

    ## https://scikit-learn.org/stable/auto_examples/applications/plot_species_distribution_modeling.html#sphx-glr-auto-examples-applications-plot-species-distribution-modeling-py

        # https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html



##### 2.7.2.1. Ampliando o SVM de uma classe 

    # Uma versão linear online do One-Class SVM é implementada em linear_model.SGDOneClassSVM. Essa implementação escala linearmente com o número de amostras e pode ser usada com uma aproximação de kernel para aproximar a solução de um svm.OneClassSVM kernelizado cuja complexidade é, na melhor das hipóteses, quadrática no número de amostras. Consulte a seção SVM de uma classe online para obter mais detalhes. 



    ## Exemplos:

    ## See One-Class SVM versus One-Class SVM using Stochastic Gradient Descent for an illustration of the approximation of a kernelized One-Class SVM with the linear_model.SGDOneClassSVM combined with kernel approximation. (https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgdocsvm_vs_ocsvm.html#sphx-glr-auto-examples-linear-model-plot-sgdocsvm-vs-ocsvm-py)