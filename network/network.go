package network
/*
 Not inspired by anyone, but was greatly helped by the videos by 3b1b and Sebastian Lague 
 on Neural Networks
 The equations/ understanding of backpropagation they gave me definitely helped, 
 as well as other information such as cost functions, activation functions etc
*/
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
	// initialise a new network
	// get number of neurons in each layer, number of layers and learning rate
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

	// keep track of activation values and weighted sums
	// needed for backpropagation
	allActivations := vector{}
	allWeightedSums := vector{}

	for k := 0; k < len(net.layers); k++ {
		currLayer := net.layers[k]
		currWeights, currBiases, currActivations := updateVectors(currLayer)
		prevActivations := inputs
		// the first layer has to use the inputs values as previous activations
		if k != 0 {
			prevActivations = net.layers[k-1].activations
		}
		for i := 0; i < len(currActivations); i++ {
			/* compute weighted sum:
			   sum(activation value from connecting node * weight value) ->
			   for all weights ->
			   add the bias afterwards */
			for j := 0; j < len(prevActivations); j++ {

				// s := fmt.Sprintf("Randomised weight %g", currWeights[i+(j*len(currActivations))])
				// io.WriteString(os.Stdout, s)
				// println("")

				wSum += currWeights[i+(j*len(currActivations))] * prevActivations[j]
			}
			wSum += currBiases[i]
			allWeightedSums = append(allWeightedSums, wSum)

			// w := fmt.Sprintf("w sum %g", wSum)
			// io.WriteString(os.Stdout, w)
			// println("")

			// apply activation function (sigmoid) to the weighted sum
			currActivations[i] = Activation(wSum)
			allActivations = append(allActivations, currActivations[i])
			wSum = 0.0
		}
	}
	return allActivations, allWeightedSums
}

func GetCostDerivatives(lastLayer Layer, expected []float64) vector {
	// get the partial derivatives of cost function w.r.t activation value
	costDerivatives := vector{}
	for i := 0; i < len(lastLayer.activations); i++ {
		costDerivatives = append(costDerivatives, CostDerivative(lastLayer.activations[i], expected[i]))
	}
	return costDerivatives
}

func GetActivationDerivatives(net *Network, weightedSums vector) vector {
	// get the partial derivatives of each activation value w.r.t weighted sum
	activationDerivatives := vector{}

	// keep track of how many nodes have been visited
	activationCounter := 0
	for i := 0; i < net.numLayers-1; i++ {
		currLayer := net.layers[i]
		for j := 0; j < len(currLayer.activations); j++ {
			weightedSum := weightedSums[activationCounter]
			activationDerivatives = append(activationDerivatives, ActivationDerivative(weightedSum))
			activationCounter++
		}
	}
	return activationDerivatives
}

func BackPropagation(net *Network, inputs, expected []float64) (vector, vector){
	// A LOT OF COMMENTS HERE, because the function was hard to write
	/*compute backpropagation
	go through network, find out how much each weight and bias
	has an effect on the network has a whole*/

	allActivations, allWeightedSums := FeedForward(net, inputs)

	// calculate the partial derivatives of cost func w.r.t activations
	lastLayer := net.layers[net.numLayers-2]
	costDerivatives := GetCostDerivatives(lastLayer, expected)

	// calculate the partial derivative of activation value w.r.t weighted sum
	activationDerivatives := GetActivationDerivatives(net, allWeightedSums)

	// keep track of gradients for each weight and bias
	costGradientW := vector{}
	costGradientB := vector{}

	// keep track of what node we are visiting (going from the back)
	currNode := len(allActivations) - 1
	println(currNode)

	// calculate the partial derivatives of cost w.r.t to weights, biases
	for currLayer := net.numLayers - 2; currLayer >= 0; currLayer-- {
		currWeights := net.layers[currLayer].weights

		// how many nodes the weights are connected to.
		numOutputNodes := net.layers[currLayer].nodesOut

		/* find the previous layers activations (for use in derivative)
		   for the first layer, the activations are the input
		   otherwise make the activations the ones from the previous layer*/
		prevLayerActivations := inputs
		if(currLayer != 0){
			prevLayerActivations = net.layers[currLayer - 1].activations
		}

		/* the node position decreases by the number of nodes outgoing
		has to point to the first node in the connected layer
		--the ones the weights are connected to */
		currNode -= numOutputNodes

		for j := 0; j < len(currWeights); j++ {
			// create a temp var that stores the pos of the current node in the connecting layer
			// in brackets just to be safe ;)
			tempCurrNode := (j % numOutputNodes)
			println("tempCurrNode", tempCurrNode)

			/* keep track of derivative value
			   if not the last layer, then reuse the derivative from the last layer*/
			currWeightDerivative := 1.0
			/* multiply by the derivative of cost function w.r.t to activation value
			of node connected to weight */
			currWeightDerivative *= costDerivatives[tempCurrNode]
			if(currLayer != net.numLayers - 2){
				currWeightDerivative = 1.0
				/* get the derivative of the each weight connected to the node connected to right now
				   add them up and multiply it by x*/
				
				// find number of weights (for the particular node) in the layer in front
				numNodesInFront := net.layers[currLayer + 1].nodesOut

				// add up the derivates for the weights of the nodes in front
				for weightNum := 0; weightNum < numNodesInFront; weightNum++{
					/* the derivative has the current activation value multiplied,
					   so divide it
					   add current derivative with derivative of weight in front
					   divide by the derivative of the current activation values
					   multiply by P derivative of weighted sum w.r.t activation value
					   which is just the weight value */
					weightDerivativeAddition := costGradientW[weightNum * tempCurrNode] 
					weightDerivativeAddition /= net.layers[currLayer].activations[tempCurrNode]
					weightDerivativeAddition *= net.layers[currLayer].weights[weightNum * tempCurrNode]
					currWeightDerivative += (weightDerivativeAddition)
				}

			}
			// finds the neuron value from the previous layer
			// using floor division, we can find the previous layer
			numActivationPrevLayer := j / numOutputNodes

			// derivative of weighted sum w.r.t weight is activation values from prev layer
			currWeightDerivative *= prevLayerActivations[numActivationPrevLayer]

			// multiply by the derivative of activation value w.r.t to weighted sum
			currWeightDerivative *= activationDerivatives[tempCurrNode]

			// store the value of the derivative
			costGradientW = append(costGradientW, currWeightDerivative)
		}
	}

	return costGradientW, costGradientB
}

func ApplyGradients(net *Network, costGradientW, costGradientB vector) {
	for currLayer := 0; currLayer < net.numLayers - 1; currLayer++{
		for currWeight := 0; currWeight < len(net.layers[currLayer].weights); currWeight++{
			
		}
	}
}

func Activation(weightedSum float64) float64 {
	// sigmoid function
	return 1 / (1 + math.Exp(-weightedSum))
}

func ActivationDerivative(weightedSum float64) float64 {
	// derivative of the sigmoid function, y, derivative is (y * (1 - y))
	activationValue := Activation(weightedSum)
	return activationValue * (1 - activationValue)
}

func Cost(net Network, expectedValues []int) float64 {
	// cost function, see how 'wrong' each final output value was
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
