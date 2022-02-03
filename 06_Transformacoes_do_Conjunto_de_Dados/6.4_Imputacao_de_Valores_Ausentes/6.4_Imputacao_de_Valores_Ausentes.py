########## 6.4. Imputação de valores ausentes ##########



    # Por vários motivos, muitos conjuntos de dados do mundo real contêm valores ausentes, geralmente codificados como espaços em branco, NaNs ou outros espaços reservados. Esses conjuntos de dados, no entanto, são incompatíveis com os estimadores scikit-learn, que assumem que todos os valores em uma matriz são numéricos e que todos têm e mantêm significado. Uma estratégia básica para usar conjuntos de dados incompletos é descartar linhas e/ou colunas inteiras contendo valores ausentes. No entanto, isso tem o preço de perder dados que podem ser valiosos (mesmo que incompletos). Uma estratégia melhor é imputar os valores ausentes, ou seja, inferi-los a partir da parte conhecida dos dados. Consulte a entrada do Glossário de Termos Comuns e Elementos da API sobre imputação. 
