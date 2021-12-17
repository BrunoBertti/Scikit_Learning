########## 1.10.8. Poda de custo-complexidade mínima  ##########

    # A poda de custo-complexidade mínima é um algoritmo usado para podar uma árvore para evitar o sobreajuste, descrito no Capítulo 3 de [BRE]. Este algoritmo é parametrizado por \ alpha \ ge0 conhecido como parâmetro de complexidade. O parâmetro de complexidade é usado para definir a medida de complexidade de custo, R_ \ alpha (T) de uma determinada árvore T:


        # R_ \ alpha (T) = R (T) + \ alpha | \ widetilde {T} |


    # onde | \ widetilde {T} | é o número de nós terminais em T e R (T) é tradicionalmente definido como a taxa total de classificação incorreta dos nós terminais. Como alternativa, o scikit-learn usa a impureza ponderada da amostra total dos nós terminais para R (T). Conforme mostrado acima, a impureza de um nó depende do critério. A poda de custo-complexidade mínima encontra a subárvore de T que minimiza R_ \ alpha (T).


    # A medida de complexidade de custo de um único nó é R_ \ alpha (t) = R (t) + \ alpha. O ramo, T_t, é definido como uma árvore em que o nó t é sua raiz. Em geral, a impureza de um nó é maior do que a soma das impurezas de seus nós terminais, R (T_t) <R (t). No entanto, a medida de complexidade de custo de um nó, t, e seu ramo, T_t, podem ser iguais dependendo de \ alpha. Definimos o \ alpha efetivo de um nó como o valor onde eles são iguais, R_ \ alpha (T_t) = R_ \ alpha (t) ou \ alpha_ {eff} (t) = \ frac {R (t) -R (T_t)} {| T | -1}. Um nó não terminal com o menor valor de \ alpha_ {eff} é o elo mais fraco e será removido. Este processo para quando o \ alpha_ {eff} mínimo da árvore podada é maior do que o parâmetro ccp_alpha. 