package network

import "math/rand"

type vector []float64

type Layer struct {
	nodesIn     int
	nodesOut    int
	weights     vector
	biases      vector
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
	for i := 0; i < length; i++ {
		v[i] = 0
	}
	return v
}

func generateRandomVector(length int) vector {
	// make activations between 0 and 1
	random := vector{}
	for i := 0; i < length; i++ {
		random[i] = rand.Float64()
	}
	return random
}

func NewNetwork(layers []int, lRate float32) Network {
	var ls []Layer

	outputNodes := layers[len(layers)-1]
	inputNodes := layers[0]

	// create the output layer
	weights := generateRandomVector(layers[len(layers)-2] * outputNodes)
	biases := generateZeroVector(outputNodes)
	activations := generateZeroVector(outputNodes)
	ls[len(layers)-1] = NewLayer(layers[len(layers)-2], outputNodes, weights, biases, activations)

	// // input layer has 0 inputs and the number of input nodes output
	// ls[0] = NewLayer(0, inputNodes, weights, biases, activations)

	// ls[0] is inbetween the first and second layer
	// ls[1] is inbetween the second and third layer ...
	// create a new layer for every hidden layer
	for i := 0; i < len(layers)-1; i++ {
		weights = generateRandomVector(layers[i] * layers[i+1])
		biases = generateZeroVector(layers[i+1])
		activations = generateZeroVector(layers[i+1])
		ls[i] = NewLayer(layers[i], layers[i+1], weights, biases, activations)
	}

	n := Network{inputNodes, outputNodes, len(layers), ls, lRate}
	return n
}

func activationFunction(inp float64) float64{
	if(inp > 0.0){
		return 1.0
	}
	return 0.0
}

func FeedForward(net *Network, inputs[] float64) {
	// compute layer 0 here, because otherwise there's going to be errors
	currLayer := net.layers[0]
	currWeights := currLayer.weights
	currBiases := currLayer.biases
	currActivations := currLayer.activations
	wSum := 0.0
	for i := 0; i < len(currActivations); i++{
		// compute weighted sum
		for j := 0; j < len(inputs); j++{
			wSum += currWeights[i + (j * len(currActivations))] * inputs[j]
		}
		wSum += currBiases[i]
		currActivations[i] = activationFunction(wSum)
		wSum = 0.0
	}

	// for i := 1; i < net.numLayers; i++ {
	// 	for j := 0; j < len(net.layers[i].activations); j++{
	// 		// compute weighted sum

	// 	}
	// }
}
