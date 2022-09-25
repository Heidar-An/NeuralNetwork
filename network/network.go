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
	learningRate float64
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

func NewNetwork(layers []int, lRate float64) Network {
	// initialise a new network
	// get number of neurons in each layer, number of layers and learning rate
	var networkLayers []Layer

	// ls[0] is inbetween the first and second layer
	// ls[1] is inbetween the second and third layer ...
	// create a new layer for every hidden layer
	for i := 0; i < len(layers) - 1; i++ {
		weights, biases, activations := initialiseVectors(layers[i]*layers[i + 1], layers[i + 1])
		networkLayers = append(networkLayers, NewLayer(layers[i], layers[i + 1], weights, biases, activations))
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

func calcWeightPastDerivative(net *Network, costGradientW vector, currLayer, currOutputNode int) float64 {
	// used to calculate part of the derivative from before for the weight, if the weight is not in the last layer
	currWeightDerivative := 1.0
	/* get the derivative of the each weight connected to the node connected to right now
	   add them up and multiply it by x*/
				
	// find number of weights (for the particular node) in the layer in front
	numNodesInFront := net.layers[currLayer + 1].nodesOut

	// add up the derivates for the weights of the nodes in front
	for weightNum := 0; weightNum < numNodesInFront; weightNum++{
		/* the derivative has the current activation value multiplied,
		   so divide it
		   add current derivative with derivative of weight in front
		   divide by the P derivative of the current activation values
		   multiply by P derivative of weighted sum w.r.t activation value
		   which is just the weight value */
		weightDerivativeAddition := costGradientW[weightNum * currOutputNode] 
		weightDerivativeAddition /= net.layers[currLayer].activations[currOutputNode]
		weightDerivativeAddition *= net.layers[currLayer].weights[weightNum * currOutputNode]
		currWeightDerivative += (weightDerivativeAddition)
	}
	return currWeightDerivative
}

func calcFullWeightDerivative(net *Network, costGradientW *vector, costDerivatives, prevLayerActivations, activationDerivatives vector, currOutputNode, currInputNode, currLayerIndex int){
	println("outputNode", currOutputNode)

	/* keep track of derivative value
	   if not the last layer, then reuse the derivative from the last layer*/
	currWeightDerivative := 1.0
	
	if(currLayerIndex == net.numLayers - 2){
		/* multiply by the derivative of cost function w.r.t to activation value
		of node connected to weight */
		currWeightDerivative *= costDerivatives[currOutputNode]
	}else{
		// calculate the part of the derivative for the current weight if not in the last layer
		currWeightDerivative = calcWeightPastDerivative(net, *(costGradientW), currLayerIndex, currOutputNode)
	}	
	// multiply by the derivative of activation value w.r.t to weighted sum
	currWeightDerivative *= activationDerivatives[currOutputNode]

	// derivative of weighted sum w.r.t weight is activation values from prev layer
	// the derivative is the activation value of the previous neuron
	currWeightDerivative *= prevLayerActivations[currInputNode]

	// store the value of the derivative
	*costGradientW = append(*(costGradientW), currWeightDerivative)
}

func calcFullBiasDerivative(net *Network, costGradientB *vector, costDerivatives, prevLayerActivations, activationDerivatives vector, currOutputNode, currLayerIndex int) {
	// keep track of current derivative for bias
	currBiasDerivative := 1.0

	if(currLayerIndex == net.numLayers - 2){
		/* multiply by the derivative of cost function w.r.t to activation value
		of node connected to bias */
		currBiasDerivative *= costDerivatives[currOutputNode]
	}else{
		// calculate the part of the derivative for the current bias if not in the last layer
		currBiasDerivative = calcBiasPastDerivative(net, *(costGradientB), currLayerIndex, currOutputNode)
	}
	
	// derivative of weighted sum w.r.t bias from prev layer is just 1.0
	// I KNOW THE COMPUTATION IS NOT NECESSARY, but I am leaving it here for completeness
	currBiasDerivative *= 1.0

	// multiply by the derivative of the activation value w.r.t to weighted sum
	currBiasDerivative *= activationDerivatives[currOutputNode]

	// store the value of the derivative
	*costGradientB = append(*(costGradientB), currBiasDerivative)
}

func calcBiasPastDerivative(net *Network, costGradientB vector, currLayer, currOutputNode int) float64 {
	// used to calculate part of the derivative from before for the bias, if the bias is not in the last layer
	currBiasDerivative := 1.0

	numNodesInFront := net.layers[currLayer + 1].nodesOut

	// add up the derivatives for the biases of the neurons/nodes in front
	for biasNum := 0; biasNum < numNodesInFront; biasNum++{
		/* the derivative has the the derivative of weighted w.r.t bias multiplied,
		   so divide it - not needed
		   add current derivative with derivative of bias in front
		   divide by the derivative of the current activation values - which is just 1
		   multiply by P derivative of weighted sum w.r.t activation value
		   which is just the weight value */
		biasDerivativeAddition := costGradientB[biasNum * currOutputNode] 
		biasDerivativeAddition *= net.layers[currLayer].weights[biasNum * currOutputNode]
		currBiasDerivative += (biasDerivativeAddition)
	}
	return currBiasDerivative
}

func BackPropagation(net *Network, inputs, expected []float64) (vector, vector){
	// A LOT OF COMMENTS HERE, because the function was hard to write

	/*compute backpropagation
	go through network, find out how much each weight and bias
	has an effect on the network has a whole. Objective is to minimise cost function.
	Calculate the direction of greatest change, then store it*/

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
	for currLayerIndex := net.numLayers - 2; currLayerIndex >= 0; currLayerIndex-- {
		// how many nodes the weights are connected to.
		numOutputNodes := net.layers[currLayerIndex].nodesOut
		numInputNodes := net.layers[currLayerIndex].nodesIn

		/* find the previous layers activations (for use in derivative)
		   for the first layer, the activations are the input
		   otherwise make the activations the ones from the previous layer*/
		prevLayerActivations := inputs
		if(currLayerIndex != 0){
			prevLayerActivations = net.layers[currLayerIndex - 1].activations
		}

		/* the node position decreases by the number of nodes outgoing
		has to point to the first node in the connected layer
		--the ones the weights are connected to */
		currNode -= numOutputNodes

		// outer loop run for the number of biases
		for currOutputNode := 0; currOutputNode < numOutputNodes; currOutputNode++{
			// inner loop run for the number of weights, weights = nodes in * nodes out
			for currInputNode := 0; currInputNode < numInputNodes; currInputNode++{
				// calculate and store the derivative of the weight
				calcFullWeightDerivative(net, &costGradientW, costDerivatives, 
					prevLayerActivations, activationDerivatives, currOutputNode, currInputNode,
					currLayerIndex)
			}

			// calculate and store the derivative of the bias
			calcFullBiasDerivative(net, &costGradientB, costDerivatives, 
				prevLayerActivations, activationDerivatives, currOutputNode, 
				currLayerIndex)
		}

	}

	return costGradientW, costGradientB
}

func ApplyGradients(net *Network, costGradientW, costGradientB vector) {
	// keep track of how many weights and biases have been visited
	totalWeightCounter := 0
	totalBiasCounter := 0

	// because our cost function did actualValue - expectedValue, we have to subtract the weight
	for currLayerIndex := 0; currLayerIndex < net.numLayers - 1; currLayerIndex ++{
		for currWeightIndex := 0; currWeightIndex < len(net.layers[currLayerIndex].weights); currWeightIndex ++{
			currWeight := net.layers[currLayerIndex].weights[currWeightIndex]
			currWeight += net.learningRate * costGradientW[totalWeightCounter]
			totalWeightCounter++
		}

		for currBiasIndex := 0; currBiasIndex < len(net.layers[currLayerIndex].biases); currBiasIndex ++{
			currBias := net.layers[currLayerIndex].biases[currBiasIndex]
			currBias -= net.learningRate * costGradientB[totalWeightCounter]
			totalBiasCounter++
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

func Cost(net Network, expectedValues []float64) float64 {
	// cost function, see how 'wrong' each final output value was
	// calculate and return (expected value - actual value) ** 2
	lastLayerActivations := net.layers[net.numLayers-1].activations
	totalCost := 0.0
	for i := 0; i < len(lastLayerActivations); i++ {
		cost := (lastLayerActivations[i] - expectedValues[i])
		totalCost += cost * cost
	}

	return totalCost
}

func CostDerivative(outputActivation, expectedValue float64) float64 {
	// partial derivative of cost with respect to the activation of an output node
	return 2 * (outputActivation - expectedValue)
}
