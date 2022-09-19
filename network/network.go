package network

import "math/rand"

type vector []float64

type Layer struct {
	nodesIn  int
	nodesOut int
	weights  vector
	biases   vector
	activations vector
}

func NewLayer(nodesIn, nodesOut int, weights, biases, activations vector) Layer {
	l := Layer{nodesIn, nodesOut, weights, biases, activations}
	return l
}

type Network struct {
	inputNodes   int
	outputNodes  int
	numLayers    int
	layers       []Layer
	learningRate float32
}

func generateZeroVector(length int) vector {
	v := vector{}
	for i := 0; i < length; i++{
		v[i] = 0
	}
	return v
}

func generateRandomVector(length int) vector {
	random := vector{}
	for i := 0; i < length; i++{
		random[i] = (rand.Float64() - 0.5) * 6
	}
	return random
}

func NewNetwork(layers []int, lRate float32) Network {
	var ls []Layer

	outputNodes := layers[len(layers) - 1]
	inputNodes := layers[0]

	// // input layer has 0 inputs and the number of input nodes output
	// ls[0] = NewLayer(0, inputNodes, weights, biases, activations)

	// create a new layer for every hidden layer
	for i := 1; i < len(layers) - 1; i++ {
		weights := generateRandomVector(ls[i - 1].nodesOut * layers[i])
		biases := generateZeroVector(layers[i])
		activations := generateZeroVector(layers[i])
		ls[i] = NewLayer(ls[i - 1].nodesOut, layers[i], weights, biases, activations)
	}

	// create the output layer
	weights := generateRandomVector(ls[len(layers) - 2].nodesOut * outputNodes)
	biases = generateZeroVector(outputNodes)
	activations = generateZeroVector(outputNodes)
	ls[len(layers) - 1] = NewLayer(ls[len(layers) - 2].nodesOut, outputNodes, weights, biases, activations)

	n := Network{inputNodes, outputNodes, len(layers), ls, lRate}
	return n
}

func FeedForward(net &Network){

}
