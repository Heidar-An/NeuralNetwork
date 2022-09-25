package main

import (
	"NeuralNetwork/network"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

type Game struct{}

func (g *Game) Update() error{
	if(ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft)){
		LeftButtonClicked()
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	ebitenutil.DebugPrint(screen, "Hello, World!")
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 560, 560
}

func mainT(){
	//game := &Game{}
	ebiten.SetWindowSize(560, 560)
	ebiten.SetWindowTitle("Digit Drawer")
	if err := ebiten.RunGame(&Game{}); err != nil {
		panic(err)
	}
}

func LeftButtonClicked(){
	println("Clicked.")

	x, y := ebiten.CursorPosition()
	println(x, y)
}

func main() {
	layers := []int{2, 2, 2}
	learningRate := 0.1
	NNetwork := network.NewNetwork(layers, learningRate)
	inputs := []float64{2.5, 1}
	expected := []float64{1.0, 1.0}

	for i := 0; i < 50; i++{
		costGradientW, costGradientB := network.BackPropagation(&NNetwork, inputs, expected)

		network.ApplyGradients(&NNetwork, costGradientW, costGradientB)
	}
}
