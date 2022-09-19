package network

import (
	"math"
	"math/rand"
	"fmt"
	"io"
	"os"
)

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

func OutputActivationValues(NNetwork *Network){
	for i := 0; i < len(NNetwork.layers); i++{
		currLayer := NNetwork.layers[i]
		currActivations := currLayer.activations
		for j := 0; j < len(currActivations); j++{
			println("Activation", currActivations[j])
		}
	}
}

func generateZeroVector(length int) vector {
	zeroVector := vector{}
	for i := 0; i < length; i++ {
		zeroVector = append(zeroVector, 0)
	}
	return zeroVector
}

func generateRandomVector(length int) vector {
	// make activations between 0 and 1
	randomVector := vector{}
	for i := 0; i < length; i++ {
		randomVector = append(randomVector, (rand.Float64() - 0.5) * 2)
	}
	return randomVector
}

func createVectors(numWeights, numOther int) (vector, vector, vector) {
	weights := generateRandomVector(numWeights)
	biases := generateZeroVector(numOther)
	activations := generateZeroVector(numOther)
	return weights, biases, activations
}

func NewNetwork(layers []int, lRate float32) Network {
	var networkLayers []Layer

	// ls[0] is inbetween the first and second layer
	// ls[1] is inbetween the second and third layer ...
	// create a new layer for every hidden layer
	for i := 0; i < len(layers) - 1; i++ {
		weights, biases, activations := createVectors(layers[i] * layers[i + 1], layers[i + 1])
		networkLayers = append(networkLayers, NewLayer(layers[i], layers[i + 1], weights, biases, activations))
	}

	n := Network{layers[0], layers[len(layers) - 1], len(layers), networkLayers, lRate}
	return n
}

func activationFunction(inp float64) float64{
	// sigmoid function
	return 1 / (1 + math.Exp(-inp))
}

func FeedForward(net *Network, inputs[] float64) {
	// compute layer 0 here, because otherwise there's going to be errors
	
	wSum := 0.0
	
	for k := 0; k < len(net.layers); k++{
		currLayer := net.layers[k]
		currWeights := currLayer.weights
		currBiases := currLayer.biases
		currActivations := currLayer.activations

		prevActivations := inputs
		if(k != 0){
			prevActivations = net.layers[k - 1].activations
		}
		for i := 0; i < len(currActivations); i++{
			// compute weighted sum
			for j := 0; j < len(prevActivations); j++{
				s := fmt.Sprintf("Randomised weight %g", currWeights[i + (j * len(currActivations))])
				io.WriteString(os.Stdout, s)
				println("")
				wSum += currWeights[i + (j * len(currActivations))] * prevActivations[j]
			}
			wSum += currBiases[i]
			w := fmt.Sprintf("w sum %g", wSum)
			io.WriteString(os.Stdout, w)
			println("")
			currActivations[i] = activationFunction(wSum)
			wSum = 0.0
		}
	}
}

func ApplyGradients(net *Network){

}

func CostDerivative(outputActivation, expectedValue float64) float64{
	// partial derivative of cost with respect to the activation of an output node
	return 2 * (outputActivation - expectedValue)
}

func CalCost(net Network, expectedValues []int) float64{
	// cost function
	// calculate and return (expected value - actual value) ** 2
	lastLayerActivations := net.layers[net.numLayers - 1].activations
	totalCost := 0.0
	for i := 0; i < len(lastLayerActivations); i++{
		cost := (lastLayerActivations[i] - float64(expectedValues[i]))
		totalCost += cost * cost
	}

	return totalCost
}
