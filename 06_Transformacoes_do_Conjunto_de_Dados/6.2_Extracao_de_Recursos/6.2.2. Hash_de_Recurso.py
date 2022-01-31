##########  6.2.2. Hash de recurso ##########

    # A classe FeatureHasher é um vetorizador de alta velocidade e pouca memória que usa uma técnica conhecida como hashing de recursos, ou o “truque de hash”. Em vez de construir uma tabela de hash dos recursos encontrados no treinamento, como fazem os vetorizadores, as instâncias de FeatureHasher aplicam uma função de hash aos recursos para determinar seu índice de coluna diretamente nas matrizes de amostra. O resultado é o aumento da velocidade e a redução do uso de memória, em detrimento da capacidade de inspeção; o hasher não lembra como eram os recursos de entrada e não possui o método inverse_transform.

    # Como a função de hash pode causar colisões entre recursos (não relacionados), uma função de hash com sinal é usada e o sinal do valor de hash determina o sinal do valor armazenado na matriz de saída para um recurso. Dessa forma, é provável que as colisões cancelem em vez de acumular erros, e a média esperada do valor de qualquer recurso de saída é zero. Esse mecanismo é habilitado por padrão com alternate_sign=True e é particularmente útil para tamanhos de tabela de hash pequenos (n_features < 10000). Para grandes tamanhos de tabela de hash, ele pode ser desabilitado, para permitir que a saída seja passada para estimadores como MultinomialNB ou seletores de recursos chi2 que esperam entradas não negativas.

    # FeatureHasher aceita mapeamentos (como dict do Python e suas variantes no módulo de coleções), pares (recurso, valor) ou strings, dependendo do parâmetro do construtor input_type. Os mapeamentos são tratados como listas de pares (feature, value), enquanto as strings simples têm um valor implícito de 1, então ['feat1', 'feat2', 'feat3'] é interpretado como [('feat1', 1), ( 'faça2', 1), ('faça3', 1)]. Se um único recurso ocorrer várias vezes em uma amostra, os valores associados serão somados (assim ('feat', 2) e ('feat', 3.5) se tornarão ('feat', 5.5)). A saída do FeatureHasher é sempre uma matriz scipy.sparse no formato CSR.

    # O hashing de recursos pode ser empregado na classificação de documentos, mas, diferentemente do CountVectorizer, o FeatureHasher não faz divisão de palavras ou qualquer outro pré-processamento, exceto a codificação Unicode-to-UTF-8; veja Vetorizando um corpus de texto grande com o truque de hash, abaixo, para um tokenizer/hasher combinado.

    # Como exemplo, considere uma tarefa de processamento de linguagem natural em nível de palavra que precisa de recursos extraídos de pares (token, part_of_speech). Pode-se usar uma função geradora do Python para extrair recursos: 

def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)

    # Então, o raw_X a ser alimentado em FeatureHasher.transform pode ser construído usando: 

# raw_X = (token_features(tok, pos_tagger(tok)) for tok in corpus)


    # e alimentado a um hasher com: 

# hasher = FeatureHasher(input_type='string')
# X = hasher.transform(raw_X)


    # para obter uma matriz X scipy.sparse.

    # Observe o uso de uma compreensão do gerador, que introduz preguiça na extração de recursos: os tokens são processados apenas sob demanda do hasher. 



##### 6.2.2.1. Detalhes de implementação 

    # FeatureHasher usa a variante assinada de 32 bits de MurmurHash3. Como resultado (e devido a limitações no scipy.sparse), o número máximo de recursos suportados atualmente é 2^{31} - 1.

    # A formulação original do truque de hashing por Weinberger et al. usou duas funções de hash separadas h e \xi para determinar o índice da coluna e o sinal de um recurso, respectivamente. A presente implementação funciona sob a suposição de que o bit de sinal de MurmurHash3 é independente de seus outros bits.

    # Como um módulo simples é usado para transformar a função hash em um índice de coluna, é aconselhável usar uma potência de dois como parâmetro n_features; caso contrário, os recursos não serão mapeados uniformemente para as colunas. 





    ## Referências:

    ## Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola and Josh Attenberg (2009). Feature hashing for large scale multitask learning. Proc. ICML. (https://alex.smola.org/papers/2009/Weinbergeretal09.pdf)

    ## https://alex.smola.org/papers/2009/Weinbergeretal09.pdf