package main

import (
	"NeuralNetwork/network"
)

func main() {
	layers := []int{2, 2, 2}
	learningRate := 0.1
	currNetwork := network.NewNetwork(layers, float32(learningRate))
	inputs := []float64{2.5, 1}
	
	network.FeedForward(&currNetwork, inputs)
}
