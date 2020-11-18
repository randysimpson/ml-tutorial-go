package main

import (
	"k8s.io/klog"
	"math"
	"math/rand"
	"github.com/randysimpson/go-matrix/matrix"
	"bufio"
	"os"
	"strings"
	"strconv"
	"fmt"
)

func ReadCSV(filename string, beginColumn int, endColumn int) ([][]float64, error) {
	var result [][]float64

	file, err := os.Open(filename)
	if err != nil {
		return result, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		//split the line into columns based on the comma
		columns := strings.Split(scanner.Text(), ",")
		
		var data []float64

		//only get columns desired.
		for _, v := range columns[beginColumn:endColumn + 1] {
			//convert from string to float64
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return result, err
			}

			data = append(data,f)
		}
		result = append(result, data)
	}

	if err := scanner.Err(); err != nil {
		return result, err
	}

	return result, nil
}

func UniqueRandomSlice(low int, high int, count int) []int {
	var result []int

	for i := 0; i < count; i++ {
		//ensure we get unique number.
		isUnique := false
		val := -1
		for !isUnique {
			isUnique = true
			val = rand.Intn(high - low) + low
			for _, v := range result {
				if v == val {
					isUnique = false
					break
				}
			}
		}
		result = append(result, val)
	}

	return result
}

func MeanByColumn(slice [][]float64) []float64 {
	rowCount := len(slice)
	colLengh := len(slice[0])
	var sums []float64
	//initialize to zeros
	for c := 0; c < colLengh; c++ {
		sums = append(sums, 0.0)
	}
	for r := 0; r < rowCount; r++ {
		for c := 0; c < colLengh; c++ {
			sums[c] += slice[r][c]
		}
	}
	for c := 0; c < colLengh; c++ {
		sums[c] = sums[c] / float64(rowCount)
	}
	return sums
}

func StdDevByColumn(slice [][]float64) []float64 {
	rowCount := len(slice)
	colLengh := len(slice[0])
	var result []float64
	//find mean by column
	means := MeanByColumn(slice)
	//initialize to zeros
	for c := 0; c < colLengh; c++ {
		result = append(result, 0.0)
	}
	for r := 0; r < rowCount; r++ {
		for c := 0; c < colLengh; c++ {
			result[c] += math.Pow(slice[r][c] - means[c], 2.0)
		}
	}
	for c := 0; c < colLengh; c++ {
		result[c] = math.Sqrt(result[c] / float64(rowCount))
	}
	return result
}

func Sigmoid(slice [][]float64) [][]float64 {
	klog.Infof("1 + np.exp(-s)= %v\n", matrix.AddScalar(Exp(matrix.MultiplyScalar(slice, -1.0)), 1.0))
	result := Divide(1, matrix.AddScalar(Exp(matrix.MultiplyScalar(slice, -1.0)), 1.0))
	//result := matrix.MultiplyScalar(matrix.AddScalar(Exp(matrix.MultiplyScalar(slice, -1.0)), 1.0), 0.01)
	return result
}

//Compute hyperbolic tangent element-wise
func Tanh(slice [][]float64) [][]float64 {
	var result [][]float64
	for r := 0; r < len(slice); r++ {
		for c := 0; c < len(slice[r]); c++ {
			//sinh 1/2 * (np.exp(x) - np.exp(-x))
			val := 0.5 * (math.Exp(slice[r][c]) - math.Exp(-1 * slice[r][c]))
			//cosh 1/2 * (np.exp(x) + np.exp(-x))
			val = val / (0.5 * (math.Exp(slice[r][c]) + math.Exp(-1 * slice[r][c])))

			//put into results
			if len(result) - 1 < r {
				var inner = []float64{val}
				result = append(result, inner)
			} else {
				result[r] = append(result[r], val)
			}
		}
	}
	return result
}

func DerivTanh(slice [][]float64) [][]float64 {
	var result [][]float64
	for r := 0; r < len(slice); r++ {
		for c := 0; c < len(slice[r]); c++ {
			val := 1 - slice[r][c] * slice[r][c]

			//put into results
			if len(result) - 1 < r {
				var inner = []float64{val}
				result = append(result, inner)
			} else {
				result[r] = append(result[r], val)
			}
		}
	}
	return result
}

func Deriv(slice [][]float64) [][]float64 {
	var result [][]float64
	for r := 0; r < len(slice); r++ {
		for c := 0; c < len(slice[r]); c++ {
			val := slice[r][c] * (1 - slice[r][c])

			//put into results
			if len(result) - 1 < r {
				var inner = []float64{val}
				result = append(result, inner)
			} else {
				result[r] = append(result[r], val)
			}
		}
	}
	return result
}

func Exp(slice [][]float64) [][]float64 {
	var result [][]float64
	for r := 0; r < len(slice); r++ {
		for c := 0; c < len(slice[r]); c++ {
			val := math.Exp(slice[r][c])

			//put into results
			if len(result) - 1 < r {
				var inner = []float64{val}
				result = append(result, inner)
			} else {
				result[r] = append(result[r], val)
			}
		}
	}
	return result
}

func Divide(x float64, slice [][]float64) [][]float64 {
	var result [][]float64
	for r := 0; r < len(slice); r++ {
		for c := 0; c < len(slice[r]); c++ {
			val := x / slice[r][c]

			//put into results
			if len(result) - 1 < r {
				var inner = []float64{val}
				result = append(result, inner)
			} else {
				result[r] = append(result[r], val)
			}
		}
	}
	return result
}

type model struct {
	w [][]float64
	means []float64
	stds []float64
}

func Train(X [][]float64, T [][]float64) model {
	//get standardized data info
	xMeans := MeanByColumn(X)
	xStds := StdDevByColumn(X)

	//standardize data, except for the 1st column...
	var XStdTrain [][]float64
	for r := 0; r < len(X); r++ {
		var rowData []float64
		//1st column is still 1
		rowData = append(rowData, 1.0)
		for c := 1; c < len(X[0]); c++ {
			val := (X[r][c] - xMeans[c]) / xStds[c]
			rowData = append(rowData, val)
		}
		XStdTrain = append(XStdTrain, rowData)
	}

	rho := 0.1
	epoch := 500

	//setup weight matrix as initially all zeros.
	var w [][]float64
	for i := 0; i < len(XStdTrain[0]); i++ { //length of row in X
		var sliceFloat []float64
		for j := 0; j < len(T[0]); j++ { //length of row in T
			sliceFloat = append(sliceFloat, 0.0)
		}
		w = append(w, sliceFloat)
	}

	for i := 0; i < epoch; i++ {
		for j := 0; j < len(XStdTrain); j++ {
			//get the x values as a slice of slices, currently it's just a single slice.
			var xMatrix [][]float64
			xMatrix = append(xMatrix, XStdTrain[j])
			
			//put the target values into slice of slices, currently this is single slice as well.
			var tMatrix [][]float64
			tMatrix = append(tMatrix, T[j])

			//multiply X matrix by weight matrix to get predicted value for y using model
			y := matrix.Multiply(xMatrix, w)

			//put y through sigmoid function
			//yn := Sigmoid(y)
			yn := Tanh(y)

			//Find the amount of change to apply to the weight matrix
			//change := matrix.Multiply(matrix.Multiply(matrix.MultiplyScalar(matrix.Transpose(xMatrix), rho), matrix.Subtract(tMatrix, yn)), Deriv(yn))
			change := matrix.Multiply(matrix.Multiply(matrix.MultiplyScalar(matrix.Transpose(xMatrix), rho), matrix.Subtract(tMatrix, yn)), DerivTanh(yn))

			//add the change to the weight matrix
			w = matrix.Add(w, change)
		}
	}

	model := model{
		w,
		xMeans,
		xStds,
	}

	return model
}

func Use(model model, X [][]float64) [][]float64 {
	//standardize the data.
	var XStd [][]float64
	for r := 0; r < len(X); r++ {
		var rowData []float64
		//1st column is still 1
		rowData = append(rowData, 1.0)
		for c := 1; c < len(X[0]); c++ {
			val := (X[r][c] - model.means[c]) / model.stds[c]
			rowData = append(rowData, val)
		}
		XStd = append(XStd, rowData)
	}

	//run the model against the standardized values
	predicted := matrix.Multiply(XStd, model.w)

	return predicted
}

func main() {
	klog.Infoln("Initializing ml tutorial application")

	sampleSize := 20
	var X [][]float64
	var T [][]float64
	//noise := [...]float64{0.31680335, -0.14948949, 0.38754655, 0.00081273, 0.19795277,	0.00663525,	0.33078164,	0.32192327,	0.08064627,	0.03071686,	0.18712232,	-0.06178398, -0.18698076, -0.22981807, 0.08204528, -0.18790643,	0.05959775,	0.34716072,	0.13218613,	-0.23696741}
	t := [...]float64{0.28915671, 0.21653494, 0.25185551, 0.01140274, 0.48967839, -0.08865728, 0.49066828, 0.08029735, 0.07973087, 0.79336589, 0.43872394, 0.67417021, 0.40204166, 0.43322684, 1.0246913, 0.81918133, 0.72089012, 1.16564833, 0.88999126, 1.36328895}

	//build X and T
	for i := 0; i < sampleSize; i++ {
		xVal := float64(i) * 0.52631579
		var xSlice []float64
		xSlice = append(xSlice, 1.0)
		xSlice = append(xSlice, xVal)
		X = append(X, xSlice)

		var tSlice []float64
		//tSlice = append(tSlice, xVal * 0.1 + noise[i])
		tSlice = append(tSlice, t[i])
		T = append(T, tSlice)
	}
	
	klog.Infof("X: =%v\n", X)
	klog.Infof("T: =%v\n", T)

	fmt.Println()
	fmt.Println("graph")
	for i := 0; i < len(X); i++ {
		fmt.Printf("{x: %v, y: %v},", X[i][1], T[i][0])
	}
	fmt.Println()

	//setup weight matrix as initially all zeros.
	var w [][]float64
	for i := 0; i < len(X[0]); i++ { //length of row in X
		var sliceFloat []float64
		for j := 0; j < len(T[0]); j++ { //length of row in T
			sliceFloat = append(sliceFloat, 0.0)
		}
		w = append(w, sliceFloat)
	}

	rho := 0.1

	for i := 0; i < 100; i++ {
		for j := 0; j < len(X); j++ {
			//get the x values as a slice of slices, currently it's just a single slice.
			var xMatrix [][]float64
			xMatrix = append(xMatrix, X[j])
			klog.Infof("xMatrix: =%v\n", xMatrix)

			//put the target values into slice of slices, currently this is single slice as well.
			var tMatrix [][]float64
			tMatrix = append(tMatrix, T[j])
			klog.Infof("tMatrix: =%v\n", tMatrix)

			klog.Infof("x @ w: =%v\n", matrix.Multiply(xMatrix, w))

			//put y through sigmoid function
			yn := Sigmoid(matrix.Multiply(xMatrix, w))
			klog.Infof("yn: =%v\n", yn)
			//yn := Tanh(y)

			klog.Infof("x.T * rho: =%v\n", matrix.MultiplyScalar(matrix.Transpose(xMatrix), rho))
			klog.Infof("tn - yn: =%v\n", matrix.Subtract(tMatrix, yn))
			klog.Infof("df: =%v\n", Deriv(yn))
			klog.Infof("w: =%v\n", matrix.Multiply(matrix.Multiply(matrix.MultiplyScalar(matrix.Transpose(xMatrix), rho), matrix.Subtract(tMatrix, yn)), Deriv(yn)))
			w = matrix.Add(w, matrix.Multiply(matrix.Multiply(matrix.MultiplyScalar(matrix.Transpose(xMatrix), rho), matrix.Subtract(tMatrix, yn)), Deriv(yn)))
		}
	}

	klog.Infof("w: =%v\n", w)

	prediction := Sigmoid(matrix.Multiply(X, w))

	klog.Infof("prediction: =%v\n", prediction)
	fmt.Println()
	fmt.Println("graph")
	for i := 0; i < len(prediction); i++ {
		fmt.Printf("{x: %v, y: %v},", X[i][1], prediction[i][0])
	}
}