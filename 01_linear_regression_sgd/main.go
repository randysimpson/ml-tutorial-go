package main

import (
	"k8s.io/klog"
	"math/rand"
	"math"
	"github.com/randysimpson/go-matrix/matrix"
)

func main() {
	klog.Infoln("Initializing ml tutorial application");

	sampleSize := 100
	var X []float64
	var T []float64

	//build the X and T array of sample data
	for i := 0; i < sampleSize; i++ {
		//get random value from 0 - 10
		value := rand.Float64() * 10.0

		//add this value to the X array
		X = append(X, value)

		//get a fraction to add some error to the equation to simulate an error on real data.
		noise := rand.Float64() / 10.0
		//add the corresponding T after using the function
		tValue := 2.0 - 0.1 * value + 0.05 * math.Pow(value - 7.0, 2) + noise
		T = append(T, tValue)
	}

	klog.Infof("X: =%v\n", X)
	klog.Infof("T: =%v\n", T)

	//add 1's to our X slice, this is required so that we can apply matrix functions.
	//and place them into slice of slieces.
	var X1 [][]float64
	for i := 0; i < sampleSize; i++ {
		var arrayFloat []float64
		arrayFloat = append(arrayFloat, 1.0)
		arrayFloat = append(arrayFloat, X[i])
		X1 = append(X1, arrayFloat)
	}

	//create slice of slieces for T's
	var T1 [][]float64
	for i := 0; i < sampleSize; i++ {
		var arrayFloat []float64
		arrayFloat = append(arrayFloat, T[i])
		T1 = append(T1, arrayFloat)
	}

	learning_rate := 0.01
	epoch := 10

	//setup weight matrix as initially all zeros.
	var w [][]float64
	for i := 0; i < len(X1[0]); i++ { //length of row in X
		var arrayFloat []float64
		for j := 0; j < len(T1[0]); j++ { //length of row in T
			arrayFloat = append(arrayFloat, 0.0)
		}
		w = append(w, arrayFloat)
	}
	
	klog.Infof("learning_rate: =%v\n", learning_rate)
	klog.Infof("epoch: =%v\n", epoch)
	klog.Infof("Initial w: =%v\n", w)

	for i := 0; i < epoch; i++ {
		for j := 0; j < sampleSize; j++ {
			//get the x values as a slice of slices, currently it's just a single slice.
			var xMatrix [][]float64
			xMatrix = append(xMatrix, X1[j])

			//multiply it by weight matrix to get predicted value for y using model
			y := matrix.Multiply(xMatrix, w)
			
			//put the target values into slice of slices, currently this is single slice as well.
			var tMatrix [][]float64
			tMatrix = append(tMatrix, T1[j])

			//find the error
			err := matrix.Subtract(tMatrix, y)

			//Find the amount of change to apply to the weight matrix
			xT := matrix.Transpose(xMatrix)
			diff := matrix.Multiply(xT, err)
			change := matrix.MultiplyScalar(diff, learning_rate)

			//add the change to the weight matrix
			w = matrix.Add(w, change)
		}
	}

	klog.Infof("Final w = %v\n", w)

	//run the model against the X values
	predicted := matrix.Multiply(X1, w)
	klog.Infof("predicted y's= %v\n", predicted)
}