########## 1.17.5. Algoritmos ##########


    # O MLP treina usando Stochastic Gradient Descent, Adam ou L-BFGS. Stochastic Gradient Descent (SGD) atualiza parâmetros usando o gradiente da função de perda em relação a um parâmetro que precisa de adaptação, ou seja,


        # w \ leftarrow w - \ eta (\ alpha \ frac {\ partial R (w)} {\ partial w}
        #        + \ frac {\ perda parcial} {\ parcial w})
    

    # onde \ eta é a taxa de aprendizagem que controla o tamanho do passo na pesquisa de espaço de parâmetro. Perda é a função de perda usada para a rede.

    # Mais detalhes podem ser encontrados na documentação do SGD

    # Adam é semelhante ao SGD no sentido de que é um otimizador estocástico, mas pode ajustar automaticamente a quantidade de parâmetros de atualização com base em estimativas adaptativas de momentos de ordem inferior.

    # Com SGD ou Adam, o treinamento oferece suporte ao aprendizado online e em minilote.

    # L-BFGS é um solucionador que aproxima a matriz Hessiana que representa a derivada parcial de segunda ordem de uma função. Além disso, ele se aproxima do inverso da matriz Hessiana para realizar atualizações de parâmetros. A implementação usa a versão Scipy do L-BFGS.

    # Se o solucionador selecionado for ‘L-BFGS’, o treinamento não suporta aprendizagem online nem em minilote. 