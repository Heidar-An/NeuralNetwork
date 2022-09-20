package network

import (
	"fmt"
	"io"
	"math"
	"math/rand"
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
	// define and return a new layer
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

func OutputActivationValues(NNetwork Network) {
	for i := 0; i < len(NNetwork.layers); i++ {
		currLayer := NNetwork.layers[i]
		currActivations := currLayer.activations
		for j := 0; j < len(currActivations); j++ {
			s := fmt.Sprintf("Activation %g", currActivations[j])
			io.WriteString(os.Stdout, s)
			println("")
		}
	}
}

func generateZeroVector(length int) vector {
	// create and return a vector of size 'length' with values 0
	// used for biases and initial activations
	zeroVector := vector{}
	for i := 0; i < length; i++ {
		zeroVector = append(zeroVector, 0)
	}
	return zeroVector
}

func generateRandomVector(length int) vector {
	// create and return a vector of size 'length' with random values between -1 and 1
	// used for initial weights
	randomVector := vector{}
	for i := 0; i < length; i++ {
		randomVector = append(randomVector, (rand.Float64()-0.5)*2)
	}
	return randomVector
}

func initialiseVectors(numWeights, numOther int) (vector, vector, vector) {
	weights := generateRandomVector(numWeights)
	biases := generateZeroVector(numOther)
	activations := generateZeroVector(numOther)
	return weights, biases, activations
}

func updateVectors(currLayer Layer) (vector, vector, vector) {
	weights := currLayer.weights
	biases := currLayer.biases
	activations := currLayer.activations
	return weights, biases, activations
}

func NewNetwork(layers []int, lRate float32) Network {
	var networkLayers []Layer

	// ls[0] is inbetween the first and second layer
	// ls[1] is inbetween the second and third layer ...
	// create a new layer for every hidden layer
	for i := 0; i < len(layers)-1; i++ {
		weights, biases, activations := initialiseVectors(layers[i]*layers[i+1], layers[i+1])
		networkLayers = append(networkLayers, NewLayer(layers[i], layers[i+1], weights, biases, activations))
	}

	n := Network{layers[0], layers[len(layers)-1], len(layers), networkLayers, lRate}
	return n
}

func FeedForward(net *Network, inputs []float64) (vector, vector) {
	// completes feedforward through network
	// returns values of all the activations and weighted sums
	wSum := 0.0
	allActivations := vector{}
	allWeightedSums := vector{}

	for k := 0; k < len(net.layers); k++ {
		currLayer := net.layers[k]
		currWeights, currBiases, currActivations := updateVectors(currLayer)
		prevActivations := inputs
		if k != 0 {
			prevActivations = net.layers[k-1].activations
		}
		for i := 0; i < len(currActivations); i++ {
			// compute weighted sum
			for j := 0; j < len(prevActivations); j++ {
				s := fmt.Sprintf("Randomised weight %g", currWeights[i+(j*len(currActivations))])
				io.WriteString(os.Stdout, s)
				println("")
				wSum += currWeights[i+(j*len(currActivations))] * prevActivations[j]
			}
			wSum += currBiases[i]
			allWeightedSums = append(allWeightedSums, wSum)
			w := fmt.Sprintf("w sum %g", wSum)
			io.WriteString(os.Stdout, w)
			println("")
			currActivations[i] = Activation(wSum)
			allActivations = append(allActivations, currActivations[i])
			wSum = 0.0
		}
	}
	return allActivations, allWeightedSums
}

func BackPropagation(net *Network, inputs, expected []float64) {
	allActivations, allWeightedSums := FeedForward(net, inputs)
	costDerivatives := vector{}

	// calculate the partial derivatives of cost func w.r.t activations
	lastLayer := net.layers[net.numLayers-1]
	for i := 0; i < len(lastLayer.activations); i++ {
		costDerivatives = append(costDerivatives, CostDerivative(lastLayer.activations[i], expected[i]))
	}

	// calculate the partial dertivatives of cost w.r.t to weights,biases 
	for i := net.numLayers - 1; i >= 0; i-- {

	}
}

func Activation(weightedInput float64) float64 {
	// sigmoid function
	return 1 / (1 + math.Exp(-weightedInput))
}

func ActivationDerivative(weightedInput float64) float64 {
	// derivative of the sigmoid function, y, derivative is (y * (1 - y))
	activationValue := Activation(weightedInput)
	return activationValue * (1 - activationValue)
}

func Cost(net Network, expectedValues []int) float64 {
	// cost function
	// calculate and return (expected value - actual value) ** 2
	lastLayerActivations := net.layers[net.numLayers-1].activations
	totalCost := 0.0
	for i := 0; i < len(lastLayerActivations); i++ {
		cost := (lastLayerActivations[i] - float64(expectedValues[i]))
		totalCost += cost * cost
	}

	return totalCost
}

func CostDerivative(outputActivation, expectedValue float64) float64 {
	// partial derivative of cost with respect to the activation of an output node
	return 2 * (outputActivation - expectedValue)
}

func ApplyGradients(net *Network) {

}
