########## 2.8.1. Estimativa de densidade: histogramas ##########

    # Um histograma é uma visualização simples de dados em que as caixas são definidas e o número de pontos de dados em cada caixa é computado. Um exemplo de um histograma pode ser visto no painel superior esquerdo da seguinte figura: 



        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html


    # Um grande problema com histogramas, no entanto, é que a escolha de binning pode ter um efeito desproporcional na visualização resultante. Considere o painel superior direito da figura acima. Ele mostra um histograma sobre os mesmos dados, com as caixas deslocadas para a direita. Os resultados das duas visualizações parecem totalmente diferentes e podem levar a interpretações diferentes dos dados.

    # Intuitivamente, também se pode pensar em um histograma como uma pilha de blocos, um bloco por ponto. Empilhando os blocos no espaço de grade apropriado, recuperamos o histograma. Mas e se, em vez de empilhar os blocos em uma grade regular, centralizarmos cada bloco no ponto que ele representa e somarmos a altura total em cada local? Essa ideia leva à visualização inferior esquerda. Talvez não seja tão claro quanto um histograma, mas o fato de que os dados conduzem as localizações dos blocos significa que é uma representação muito melhor dos dados subjacentes.

    # Esta visualização é um exemplo de estimativa de densidade de kernel, neste caso com um kernel de cartola (ou seja, um bloco quadrado em cada ponto). Podemos recuperar uma distribuição mais suave usando um kernel mais suave. O gráfico inferior direito mostra uma estimativa da densidade do kernel gaussiana, em que cada ponto contribui com uma curva gaussiana para o total. O resultado é uma estimativa de densidade uniforme que é derivada dos dados e funciona como um poderoso modelo não paramétrico da distribuição de pontos. 