########## 6.1.5. Visualizando Estimadores Compostos  ##########


    # Os estimadores podem ser exibidos com uma representação HTML quando mostrados em um notebook jupyter. Isso pode ser útil para diagnosticar ou visualizar um Pipeline com muitos estimadores. Esta visualização é ativada definindo a opção display em set_config: 

from sklearn import set_config
set_config(display='diagram')   
# exibe a representação HTML em um contexto jupyter 
# column_trans


    # Um exemplo da saída HTML pode ser visto na representação HTML da seção Pipeline do Column Transformer with Mixed Types. Como alternativa, o HTML pode ser escrito em um arquivo usando estimator_html_repr: 


from sklearn.utils import estimator_html_repr
with open('my_estimator.html', 'w') as f:  
    f.write(estimator_html_repr(clf))




    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer.html#sphx-glr-auto-examples-compose-plot-column-transformer-py

    ## https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py