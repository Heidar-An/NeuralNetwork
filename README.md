# Neural Network Go Library

## What is this? 
A library written using Go. The library allows users to create a neural network with a configurable number of neurons and layers.\
The library also implements backpropgation using gradient descent to gradually reduce the cost function. \ 
I tested the neural network library against the MNIST database: http://yann.lecun.com/exdb/mnist/ , a list of handwritten digits \
Lastly, I also implemented a graphical user design so that users could draw digits themselves, and test it against the neural network. 
This was done using the ebitengine graphical library https://ebitengine.org/\

## Why I did this
A feeling of wanting to learn how to use Go, and understand more about the intuition about neural networks. \
Implementing backpropagation was especially difficult, (but rewarding!) as it required me to really understand how it works

## Technologies
Go - v1.18 \
ebiten, ebitengine - v2.4.4

## Future Plans/Improvements
I plan to add some optimisers, to make it faster \
Furthermore, I plan on augmenting it to make it a convolutional neural network. However, I might redo the project in Rust instead for that.\
