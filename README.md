# Neural Network Go Library

## What is this? 
A library written using Go. The library allows users to create a neural network with a configurable number of neurons and layers. <br />
The library also implements backpropgation using gradient descent to gradually reduce the cost function. <br />
I tested the neural network library against the MNIST database: http://yann.lecun.com/exdb/mnist/ , a list of handwritten digits <br />
Lastly, I also implemented a graphical user design so that users could draw digits themselves, and test it against the neural network. 
This was done using the ebitengine graphical library https://ebitengine.org/ <br />

## Why I did this
A feeling of wanting to learn how to use Go, and understand more about the intuition about neural networks. <br />
Implementing backpropagation was especially difficult, (but rewarding!) as it required me to really understand how it works <br />
Overall, it was a good project for an introduction to neural networks, and I hope to develop it further in the future!

## Technologies
Go - v1.18 <br />
ebiten, ebitengine - v2.4.4 <br />

## Future Plans/Improvements
I plan to add some optimisers, to make it faster such as optimising matrix multiplication for caching, concurrent computation, or SIMD. I could utilise some optimsers for gradient descent such as RMSProp and Adam too. <br />
Furthermore, I plan on augmenting it to make it a convolutional neural network. However, I might redo the project in Rust instead for that. <br />
