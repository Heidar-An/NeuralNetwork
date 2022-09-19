package main

import (
	"NeuralNetwork/network"
)

func main() {
	var layers []int
	learningRate := 0.1
	layers[0] = 1
	layers[1] = 1
	network := network.NewNetwork(layers, float32(learningRate))
	for i := 0; i < len(layers); i++{
		
	}
}
