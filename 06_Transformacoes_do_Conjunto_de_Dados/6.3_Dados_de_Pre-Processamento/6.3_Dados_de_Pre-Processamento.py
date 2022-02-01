########## 6.3. Dados de pré-processamento ##########


    # O pacote sklearn.preprocessing fornece várias funções utilitárias comuns e classes de transformadores para alterar os vetores de recursos brutos em uma representação mais adequada para os estimadores downstream.

    # Em geral, os algoritmos de aprendizado se beneficiam da padronização do conjunto de dados. Se alguns outliers estiverem presentes no conjunto, scalers ou transformadores robustos são mais apropriados. Os comportamentos dos diferentes scalers, transformadores e normalizadores em um conjunto de dados contendo discrepâncias marginais são destacados em Comparar o efeito de diferentes scalers em dados com discrepâncias.