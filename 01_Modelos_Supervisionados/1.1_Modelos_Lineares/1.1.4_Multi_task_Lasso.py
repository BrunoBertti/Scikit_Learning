########## 1.1.4 Multi-task Lasso ##########
    
    # O MultiTaskLasso é um modelo linear que estima coeficientes esparsos para problemas de regressão múltipla em conjunto: y é um array 2D, de forma (n_samples, n_tasks). A restrição é que os recursos selecionados são os mesmos para todos os problemas de regressão, também chamados de tarefas.

    # A figura a seguir compara a localização das entradas diferentes de zero na matriz de coeficiente W obtida com um Lasso simples ou um MultiTaskLasso. As estimativas Lasso produzem não zeros dispersos, enquanto os não zeros da MultiTaskLasso são colunas completas. (link da figura: https://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html)

    # Exemplos: https://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html#sphx-glr-auto-examples-linear-model-plot-multi-task-lasso-support-py

    # Matematicamente, consiste em um modelo linear treinado com uma norma mista l1 e l2 para regularização. A função objetivo a minimizar é:
        # min 1/2nsamples ||XW - Y||2fro + alpha||W||21
    
    # onde Fro indica a norma Frobenius
        # ||A||fro = raiz quadrada do somatório ij do a^2ij
    
    # e l1 l2 lê
        # ||A||21 = somatório i da raiz quadrada do somatório j de a^2 ij
    
    # A implementação na classe MultiTaskLasso usa a descida por coordenadas como o algoritmo para treinar os coeficientes. 