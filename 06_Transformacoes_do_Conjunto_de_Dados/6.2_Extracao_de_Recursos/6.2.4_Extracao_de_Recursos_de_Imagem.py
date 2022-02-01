##########  6.2.4. Extração de recursos de imagem  ##########

##### 6.2.4.1. Extração de patches

    # A função extract_patches_2d extrai patches de uma imagem armazenada como uma matriz bidimensional ou tridimensional com informações de cores ao longo do terceiro eixo. Para reconstruir uma imagem de todos os seus patches, use reconstruir_from_patches_2d. Por exemplo, vamos gerar uma imagem de 4x4 pixels com 3 canais de cores (por exemplo, no formato RGB): 

import numpy as np
from sklearn.feature_extraction import image

one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
one_image[:, :, 0]  # Canal R de uma imagem RGB falsa 


patches = image.extract_patches_2d(one_image, (2, 2), max_patches=2,
    random_state=0)
patches.shape

patches[:, :, :, 0]

patches = image.extract_patches_2d(one_image, (2, 2))
patches.shape

patches[4, :, :, 0]


    # Vamos agora tentar reconstruir a imagem original dos patches calculando a média das áreas sobrepostas: 

reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4, 3))
np.testing.assert_array_equal(one_image, reconstructed)

    # A classe PatchExtractor funciona da mesma forma que extract_patches_2d, só que suporta várias imagens como entrada. Ele é implementado como um estimador, para que possa ser usado em pipelines. Ver: 

five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
patches = image.PatchExtractor(patch_size=(2, 2)).transform(five_images)
patches.shape



##### 6.2.4.2. Gráfico de conectividade de uma imagem 


    # Vários estimadores no scikit-learn podem usar informações de conectividade entre recursos ou amostras. Por exemplo, o agrupamento Ward (agrupamento hierárquico) pode agrupar apenas os pixels vizinhos de uma imagem, formando assim manchas contíguas: 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_ward_segmentation.html

    # Para isso, os estimadores usam uma matriz de 'conectividade', fornecendo quais amostras estão conectadas.

    # A função img_to_graph retorna tal matriz de uma imagem 2D ou 3D. Da mesma forma, grid_to_graph constrói uma matriz de conectividade para imagens dada a forma dessas imagens.

    # Essas matrizes podem ser usadas para impor conectividade em estimadores que usam informações de conectividade, como clustering Ward (agrupamento hierárquico), mas também para construir kernels pré-computados ou matrizes de similaridade. 




    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_ward_segmentation.html#sphx-glr-auto-examples-cluster-plot-coin-ward-segmentation-py

    ## https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py

    ## https://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#sphx-glr-auto-examples-cluster-plot-feature-agglomeration-vs-univariate-selection-py