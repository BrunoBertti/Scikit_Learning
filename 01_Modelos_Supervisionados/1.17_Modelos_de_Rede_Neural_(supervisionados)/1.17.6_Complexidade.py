########## 1.17.6. Complexidade ##########


    # Suponha que haja n amostras de treinamento, m recursos, k camadas ocultas, cada uma contendo h neurônios - para simplificar, e o neurônios de saída. A complexidade de tempo da retropropagação é O (n \ cdot m \ cdot h ^ k \ cdot o \ cdot i), onde i é o número de iterações. Como a retropropagação tem uma alta complexidade de tempo, é aconselhável começar com um número menor de neurônios ocultos e poucas camadas ocultas para treinamento. 