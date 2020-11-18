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

func StandardizeMatrix(matrix [][]float64, means []float64, stds []float64, addOnes bool) [][]float64 {
	//standardize the data
	var result [][]float64
	for r := 0; r < len(matrix); r++ {
		var rowData []float64
		if addOnes {
			//add 1st column of 1.0
			rowData = append(rowData, 1.0)
		}
		//use standardized values
		for c := 0; c < len(matrix[0]); c++ {
			val := (matrix[r][c] - means[c]) / stds[c]
			rowData = append(rowData, val)
		}
		result = append(result, rowData)
	}
	return result
}

func UnStandardizeMatrix(matrix [][]float64, means []float64, stds []float64) [][]float64 {
	//unstandardize the data
	var result [][]float64
	for r := 0; r < len(matrix); r++ {
		var rowData []float64
		//use standardized values to reverse
		for c := 0; c < len(matrix[0]); c++ {
			val := matrix[r][c] * stds[c] + means[c]
			rowData = append(rowData, val)
		}
		result = append(result, rowData)
	}
	return result
}

func AddOnes(matrix [][]float64) [][]float64 {
	//standardize the data
	var result [][]float64
	for r := 0; r < len(matrix); r++ {
		var rowData []float64
		//add 1st column of 1.0
		rowData = append(rowData, 1.0)
		//use other values
		for c := 0; c < len(matrix[0]); c++ {
			rowData = append(rowData, matrix[r][c])
		}
		result = append(result, rowData)
	}
	return result
}

func LinearMultiply(m1 [][]float64, m2 [][]float64) [][]float64 {
	//standardize the data
	var result [][]float64
	for r := 0; r < len(m1); r++ {
		var rowData []float64
		for c := 0; c < len(m1[0]); c++ {
			rowData = append(rowData, m1[r][c] * m2[r][c])
		}
		result = append(result, rowData)
	}
	return result
}

func main() {
	klog.Infoln("Initializing ml tutorial application")

	//tmp hardcode for debug
	t := [...]float64{0.01964618, 0.39473207, 0.58879436, 0.7818568, 0.21170835, 0.12359647, 0.15132348, 0.19494277, -0.10002663, 0.35566731, 0.75071577, 0.75027211, 0.78422891, 0.65468829, 0.62449052, 0.64948523, 0.33607597, 0.58227105, 0.66509991, 1.17197958, 1.48465335, 1.51016365, 0.92276195, 1.04698734, 0.76288701, 0.74708164, 0.73798824, 0.72654176, 1.3439468, 1.60337396}
	whc :=  [...]float64{-0.08134833, -0.01684297, -0.06037356, 0.01094153, 0.09047864, -0.08316043, -0.02953931, -0.09405071, 0.03050853, -0.08275847, -0.05263153, 0.04125131, 0.04965921, 0.0064405, -0.0264456, -0.08702321, 0.05581817, -0.08405909, -0.04786889, -0.00889573, 0.06841452}
	vhc := [...]float64{-0.07695679, 0.07792022, 0.06384507, -0.09881901, -0.03971482, -0.03920179,	-0.06238014, -0.07306374, 0.07850587, 0.09684893, 0.02803186, 0.03816227, 0.05775214, 0.01016488, 0.078357, 0.01321288, 0.07753837, -0.0705239, 0.07008734,  0.07284724, -0.01174968, 0.01232262, -0.02796074, 0.06543278, 0.00094029, -0.06878808, -0.05094357, 0.06474668, 0.04302347, -0.06837467, -0.09570303, -0.07087654, 0.08785966, -0.06899788, 0.02767161, -0.08539164, 0.04172768, -0.05278111, 0.07404829, 0.08390175}

	//setup training and testing data
	n := 30
	var Xtrain [][]float64
	var Ttrain [][]float64
	var Xtest [][]float64
	var Ttest [][]float64
	for i := 0; i < n; i++ {
		xVal := (20.0 / float64(n - 1)) * float64(i) - 10.0
		var xSlice []float64
		xSlice = append(xSlice, xVal)
		Xtrain = append(Xtrain, xSlice)

		//tVal := 0.2 + 0.05 * (xVal + 10) + 0.4 * math.Sin(xVal + 10) + 0.2 * rand.Float64()
		tVal := t[i]
		var tSlice []float64
		tSlice = append(tSlice, tVal)
		Ttrain = append(Ttrain, tSlice)

		xTestVal := xVal + 0.1 * rand.Float64()
		var xTestSlice []float64
		xTestSlice = append(xTestSlice, xTestVal)
		Xtest = append(Xtest, xTestSlice)

		tTestVal := 0.2 + 0.05 * (xTestVal + 10) + 0.4 * math.Sin(xTestVal + 10) + 0.2 * rand.Float64()
		var tTestSlice []float64
		tTestSlice = append(tTestSlice, tTestVal)
		Ttest = append(Ttest, tTestSlice)
	}
	
	klog.Infof("Xtrain: =%v\n", Xtrain)
	klog.Infof("Ttrain: =%v\n", Ttrain)

	fmt.Println()
	fmt.Println("training graph")
	for i := 0; i < len(Xtrain); i++ {
		fmt.Printf("{x: %v, y: %v},", Xtrain[i][0], Ttrain[i][0])
	}
	fmt.Println()
	fmt.Println("testing graph")
	for i := 0; i < len(Xtest); i++ {
		fmt.Printf("{x: %v, y: %v},", Xtest[i][0], Ttest[i][0])
	}

	//standardize the data
	//get standardized info about the training data only
	xMeans := MeanByColumn(Xtrain)
	klog.Infof("xMeans= %v\n", xMeans)
	xStds := StdDevByColumn(Xtrain)
	klog.Infof("xStds= %v\n", xStds)
	tMeans := MeanByColumn(Ttrain)
	klog.Infof("tMeans= %v\n", tMeans)
	tStds := StdDevByColumn(Ttrain)
	klog.Infof("tStds= %v\n", tStds)

	//standardize training data
	XStdTrain := StandardizeMatrix(Xtrain, xMeans, xStds, true)
	klog.Infof("XStdTrain= %v\n", XStdTrain)

	//standardize the test data.
	XStdTest := StandardizeMatrix(Xtest, xMeans, xStds, true)
	klog.Infof("XStdTest= %v\n", XStdTest)

	TStdTrain := StandardizeMatrix(Ttrain, tMeans, tStds, false)
	TStdTest := StandardizeMatrix(Ttest, tMeans, tStds, false)
	klog.Infof("TStdTrain= %v\n", TStdTrain)
	klog.Infof("TStdTest= %v\n", TStdTest)

	//number of hidden layers
	layers := 20

	//setup weight matrix as values between -0.1 and 0.1.
	var w [][]float64
	for i := 0; i < layers + 1; i++ { //hidden layers + 1
		var sliceFloat []float64
		for j := 0; j < len(Ttrain[0]); j++ { //length of row in T
			//sliceFloat = append(sliceFloat, rand.Float64() * 0.2 - 0.1)
			sliceFloat = append(sliceFloat, whc[i])
		}
		w = append(w, sliceFloat)
	}
	klog.Infof("w= %v\n", w)

	var v [][]float64
	for i := 0; i < len(Ttrain[0]) + len(Xtrain[0]); i++ { //1 + 1
		var sliceFloat []float64
		for j := 0; j < layers; j++ { //length layers
			//sliceFloat = append(sliceFloat, rand.Float64() * 0.2 - 0.1)
			sliceFloat = append(sliceFloat, vhc[i * layers + j])
		}
		v = append(v, sliceFloat)
	}
	klog.Infof("v= %v\n", v)

	rh := 0.1
	ro := 0.01
	epochs := 100000

	for i := 0; i < epochs; i++ {
		//forward pass on training data
		Z := Tanh(matrix.Multiply(XStdTrain, v))
		Z1 := AddOnes(Z)
		Y := matrix.Multiply(Z1, w)

		err := matrix.Subtract(TStdTrain, Y)

		/*klog.Infof("work= %v\n", matrix.Multiply(err, matrix.Transpose(w[1:])))
		klog.Infof("work= %v\n", DerivTanh(Z))
		klog.Infof("work= %v\n", LinearMultiply(matrix.Multiply(err, matrix.Transpose(w[1:])), DerivTanh(Z)))
		klog.Infof("work= %v\n", matrix.MultiplyScalar(matrix.Multiply(matrix.Transpose(XStdTrain), LinearMultiply(matrix.Multiply(err, matrix.Transpose(w[1:])), DerivTanh(Z))), rh))*/

		//matrix.MultiplyScalar(matrix.Transpose(XStdTrain), rh)
		v = matrix.Add(matrix.MultiplyScalar(matrix.Multiply(matrix.Transpose(XStdTrain), LinearMultiply(matrix.Multiply(err, matrix.Transpose(w[1:])), DerivTanh(Z))), rh),v)
		w = matrix.Add(matrix.MultiplyScalar(matrix.Multiply(matrix.Transpose(Z1), err), ro), w)

		//klog.Infof("Z= %v\n", Z)
		/*klog.Infof("XStdTrain shape= %v, %v\n", len(XStdTrain), len(XStdTrain[0]))
		klog.Infof("v shape= %v, %v\n", len(v), len(v[0]))
		klog.Infof("Z shape= %v, %v\n", len(Z), len(Z[0]))
		klog.Infof("Z1 shape= %v, %v\n", len(Z1), len(Z1[0]))
		klog.Infof("Y shape= %v, %v\n", len(Y), len(Y[0]))
		klog.Infof("Y= %v\n", Y)
		klog.Infof("err= %v\n", err)
		klog.Infof("W.T= %v\n", matrix.Transpose(w))*/
	}

	klog.Infof("w= %v\n", w)
	klog.Infof("v= %v\n", v)

	//run model against test data.
	YStdTest := matrix.Multiply(AddOnes(Tanh(matrix.Multiply(XStdTest, v))), w)
	YTest := UnStandardizeMatrix(YStdTest, tMeans, tStds)

	klog.Infof("YStdTest= %v\n", YStdTest)
	klog.Infof("YTest= %v\n", YTest)
	
	//plot test data using model.
	fmt.Println()
	fmt.Println("model graph")
	for i := 0; i < len(Xtest); i++ {
		fmt.Printf("{x: %v, y: %v},", Xtest[i][0], YTest[i][0])
	}
}