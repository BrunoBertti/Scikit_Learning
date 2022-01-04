########## 2.5.4. Aprendizagem de Dicionário ##########


##### 2.5.4.1. Codificação esparsa com um dicionário pré-computado 


    # O objeto SparseCoder é um estimador que pode ser usado para transformar sinais em combinações lineares esparsas de átomos de um dicionário pré-computado fixo, como uma base de wavelet discreta. Este objeto, portanto, não implementa um método de ajuste. A transformação equivale a um problema de codificação esparso: encontrar uma representação dos dados como uma combinação linear do mínimo possível de átomos de dicionário. Todas as variações de aprendizagem de dicionário implementam os seguintes métodos de transformação, controláveis por meio do parâmetro de inicialização transform_method: 

        # Busca de correspondência ortogonal (Busca de correspondência ortogonal (OMP))

        # Regressão de ângulo mínimo (regressão de ângulo mínimo)

        # Lasso calculado por regressão de ângulo mínimo

        # Laço usando descida por coordenadas (Laço)

        # Limiar 


    # Limiar é muito rápido, mas não produz reconstruções precisas. Eles têm se mostrado úteis na literatura para tarefas de classificação. Para tarefas de reconstrução de imagem, a busca de correspondência ortogonal produz a reconstrução mais precisa e imparcial.

    # Os objetos de aprendizagem do dicionário oferecem, por meio do parâmetro split_code, a possibilidade de separar os valores positivos e negativos nos resultados da codificação esparsa. Isso é útil quando o aprendizado de dicionário é usado para extrair recursos que serão usados ​​para o aprendizado supervisionado, pois permite que o algoritmo de aprendizado atribua pesos diferentes aos carregamentos negativos de um átomo particular, desde o carregamento positivo correspondente.

    # O código de divisão para uma única amostra tem comprimento 2 * n_components e é construído usando a seguinte regra: Primeiro, o código regular de comprimento n_components é calculado. Em seguida, as primeiras entradas de n_components do split_code são preenchidas com a parte positiva do vetor de código regular. A segunda metade do código de divisão é preenchida com a parte negativa do vetor de código, apenas com um sinal positivo. Portanto, o split_code não é negativo.




    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_sparse_coding.html#sphx-glr-auto-examples-decomposition-plot-sparse-coding-py



##### 2.5.4.2. Aprendizagem de dicionário genérico 

    # O aprendizado de dicionário (DictionaryLearning) é um problema de fatoração de matriz que equivale a encontrar um dicionário (geralmente supercompleto) que terá um bom desempenho na codificação esparsa dos dados ajustados.

    # A representação de dados como combinações esparsas de átomos de um dicionário supercompleto é sugerida como a forma como o córtex visual primário dos mamíferos funciona. Conseqüentemente, a aprendizagem de dicionário aplicada em patches de imagem tem mostrado bons resultados em tarefas de processamento de imagem, como finalização, pintura interna e remoção de ruído, bem como para tarefas de reconhecimento supervisionado.

    # O aprendizado do dicionário é um problema de otimização resolvido atualizando alternativamente o código esparso, como uma solução para vários problemas do Lasso, considerando o dicionário corrigido, e então atualizando o dicionário para melhor se adequar ao código esparso. 

        # \begin{split}(U^*, V^*) = \underset{U, V}{\operatorname{arg\,min\,}} & \frac{1}{2}
        #              ||X-UV||_{\text{Fro}}^2+\alpha||U||_{1,1} \\
        #              \text{subject to } & ||V_k||_2 <= 1 \text{ for all }
        #              0 \leq k < n_{\mathrm{atoms}}\end{split}



        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html

    # ||. || _ {\ text {Fro}} representa a norma de Frobenius e ||. || _ {1,1} representa a norma da matriz de entrada, que é a soma dos valores absolutos de todas as entradas na matriz. Depois de usar esse procedimento para ajustar o dicionário, a transformação é simplesmente uma etapa de codificação esparsa que compartilha a mesma implementação com todos os objetos de aprendizagem de dicionário (consulte Codificação esparsa com um dicionário pré-computado).

    # Também é possível restringir o dicionário e / ou código para ser positivo para corresponder às restrições que podem estar presentes nos dados. Abaixo estão as faces com diferentes restrições de positividade aplicadas. Vermelho indica valores negativos, azul indica valores positivos e branco representa zeros. 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html


    # A imagem a seguir mostra a aparência de um dicionário aprendido a partir de fragmentos de imagem de pixel 4x4 extraídos de parte da imagem de um rosto de guaxinim. 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html





    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html#sphx-glr-auto-examples-decomposition-plot-image-denoising-py


    ## Referências:

    ## “Online dictionary learning for sparse coding” J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009 (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)




##### 2.5.4.3. Aprendizagem de dicionário de minilote 


    # MiniBatchDictionaryLearning implementa uma versão mais rápida, mas menos precisa do algoritmo de aprendizagem de dicionário que é mais adequado para grandes conjuntos de dados.

    # Por padrão, MiniBatchDictionaryLearning divide os dados em minilotes e otimiza de maneira online, percorrendo os minilotes para o número especificado de iterações. No entanto, no momento, ele não implementa uma condição de parada.

    # O estimador também implementa partial_fit, que atualiza o dicionário iterando apenas uma vez em um minilote. Isso pode ser usado para aprendizagem online quando os dados não estão prontamente disponíveis desde o início ou para quando os dados não cabem na memória. 


    # Clustering para aprendizagem de dicionário

    # Observe que, ao usar o aprendizado de dicionário para extrair uma representação (por exemplo, para codificação esparsa), o agrupamento pode ser um bom proxy para aprender o dicionário. Por exemplo, o estimador MiniBatchKMeans é computacionalmente eficiente e implementa o aprendizado on-line com um método partial_fit.

    # Exemplo: aprendizagem online de um dicionário de partes de rostos 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html