########## 1.5.4. Descida de gradiente estocástico para dados esparsos ##########

    # Nota: A implementação esparsa produz resultados ligeiramente diferentes da implementação densa, devido a uma taxa de aprendizado reduzida para a interceptação. Consulte os detalhes de implementação.

    # Há suporte embutido para dados esparsos fornecidos em qualquer matriz em um formato suportado por scipy.sparse. Para eficiência máxima, no entanto, use o formato de matriz CSR conforme definido em scipy.sparse.csr_matrix. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py