########## 1.17.9. Mais controle com warm_start  ##########

    # Se você deseja mais controle sobre os critérios de parada ou taxa de aprendizagem no SGD, ou deseja fazer monitoramento adicional, usar warm_start = True e max_iter = 1 e iterar você mesmo pode ser útil: 


X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
for i in range(10):
    clf.fit(X, y)
    # monitoramento / inspeção adicional 



    ## Referências:

    ## “Learning representations by back-propagating errors.” Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams.(https://www.iro.umontreal.ca/~pift6266/A06/refs/backprop_old.pdf)

    ## “Stochastic Gradient Descent” L. Bottou - Website, 2010. (https://leon.bottou.org/projects/sgd)

    ## “Backpropagation” Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen - Website, 2011. (http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm)

    ## “Efficient BackProp” Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks of the Trade 1998. (http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

    ## “Adam: A method for stochastic optimization.” Kingma, Diederik, and Jimmy Ba. arXiv preprint arXiv:1412.6980 (2014). (https://arxiv.org/pdf/1412.6980v8.pdf)