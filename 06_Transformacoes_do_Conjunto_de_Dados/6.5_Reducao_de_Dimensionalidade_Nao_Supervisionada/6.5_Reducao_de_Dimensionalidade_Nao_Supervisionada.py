########## 6.5. Redução de Dimensionalidade Não Supervisionada ##########


    # Se o número de variáveis for alto, pode ser útil reduzi-lo com uma etapa não supervisionada antes das etapas supervisionadas. Muitos dos métodos de aprendizado não supervisionado implementam um método de transformação que pode ser usado para reduzir a dimensionalidade. Abaixo, discutimos dois exemplos específicos desse padrão que são muito usados.

    # Tubulação (pipeline)

    # A redução de dados não supervisionada e o estimador supervisionado podem ser encadeados em uma única etapa. Consulte Pipeline: encadeando estimadores. 