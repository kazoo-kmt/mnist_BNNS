# mnist_BNNS
* MNIST on iOS by using BNNS. 
* Mainly referred https://github.com/paiv/mnist-bnns
* The app works well if you use the model architecture here (https://github.com/paiv/mnist-bnns/blob/master/mnistios/mnistios/MnistNet.swift), but to compare other DL frameworks, I used the model architecture on Keras tutorial (https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py). Because BNNS doesn't provide softmax function, the code used sigmoid function just because the purpose of my research is benchmarking. Therefore, the accuracy of the output by this app is not good.
* After the training, the model parameters were exported as dat files. 
