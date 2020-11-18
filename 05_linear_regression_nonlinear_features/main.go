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
	epoch := 5000

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
		//for j := 0; j < 1; j++ {
			//get the x values as a slice of slices, currently it's just a single slice.
			var xMatrix [][]float64
			xMatrix = append(xMatrix, XStdTrain[j])
			
			//put the target values into slice of slices, currently this is single slice as well.
			var tMatrix [][]float64
			tMatrix = append(tMatrix, T[j])

			//multiply X matrix by weight matrix to get predicted value for y using model
			y := matrix.Multiply(xMatrix, w)

			//put y through sigmoid function
			yn := Sigmoid(y)
			//yn := Tanh(y)

			//Find the amount of change to apply to the weight matrix
			//change := matrix.Multiply(matrix.Multiply(matrix.MultiplyScalar(matrix.Transpose(xMatrix), rho), matrix.Subtract(tMatrix, yn)), Deriv(yn))
			/*klog.Infof("Deriv(yn)= %v\n", Deriv(yn))
			klog.Infof("matrix.Subtract(tMatrix, yn)= %v\n", matrix.Subtract(tMatrix, yn))
			klog.Infof("LinearMultiply(matrix.Subtract(tMatrix, yn), Deriv(yn))= %v\n", LinearMultiply(matrix.Subtract(tMatrix, yn), Deriv(yn)))
			klog.Infof("0.25 * -5.840819300294327= %v\n", (0.25 * -5.840819300294327))
			klog.Infof("matrix.Transpose(xMatrix)= %v\n", matrix.Transpose(xMatrix))
			klog.Infof("matrix.Multiply(matrix.Transpose(xMatrix), LinearMultiply(matrix.Subtract(tMatrix, yn), Deriv(yn)))= %v\n", matrix.Multiply(matrix.Transpose(xMatrix), LinearMultiply(matrix.Subtract(tMatrix, yn), Deriv(yn))))*/
			change := matrix.MultiplyScalar(matrix.Multiply(matrix.Transpose(xMatrix), LinearMultiply(matrix.Subtract(tMatrix, yn), Deriv(yn))), rho)
			//change := matrix.Multiply(matrix.Multiply(matrix.MultiplyScalar(matrix.Transpose(xMatrix), rho), matrix.Subtract(tMatrix, yn)), DerivTanh(yn))

			//add the change to the weight matrix
			w = matrix.Add(w, change)

			/*if i == epoch - 1 && j == len(XStdTrain) - 1 {
				klog.Infof("xMatrix: =%v\n", xMatrix)
				klog.Infof("tMatrix: =%v\n", tMatrix)
				klog.Infof("yn: =%v\n", yn)
			}*/
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
	//predicted := Sigmoid(matrix.Multiply(XStd, model.w))
	predicted := matrix.Multiply(XStd, model.w)

	return predicted
}

func main() {
	klog.Infoln("Initializing ml tutorial application")

	sampleSize := 40
	var X []float64
	var T []float64

	//build the X and T array of sample data
	for i := 0; i < sampleSize; i++ {
		//get random value from 0 - 3
		value := rand.Float64() * 3.0

		//add this value to the X array
		X = append(X, value)

		//get a fraction to add some error to the equation to simulate an error on real data.
		noise := (rand.Float64() * 2.0) - 1.0
		//add the corresponding T after using the function
		tValue := -1 + 0 * value + 0.1 * math.Pow(value, 2) - 0.02 * math.Pow(value, 3) + noise
		T = append(T, tValue)
	}

	//build the X and T array of sample data
	for i := 0; i < sampleSize; i++ {
		//get random value from 0 - 3
		value := (rand.Float64() * 4.0) + 6.0

		//add this value to the X array
		X = append(X, value)

		//get a fraction to add some error to the equation to simulate an error on real data.
		noise := (rand.Float64() * 2.0) - 1.0
		//add the corresponding T after using the function
		tValue := -1 + 0 * value + 0.1 * math.Pow(value, 2) - 0.02 * math.Pow(value, 3) + noise
		T = append(T, tValue)
	}

	klog.Infof("X: =%v\n", X)
	klog.Infof("T: =%v\n", T)

	//setup matrices
	var Xdata [][]float64
	var Tdata [][]float64

	for i := 0; i < len(X); i++ {
		//setup X matrix
		var sliceX []float64
		//we will add 1 as the first item
		sliceX = append(sliceX, 1.0)
		sliceX = append(sliceX, X[i])
		sliceX = append(sliceX, math.Pow(X[i], 2))
		sliceX = append(sliceX, math.Pow(X[i], 3))
		sliceX = append(sliceX, math.Pow(X[i], 4))
		sliceX = append(sliceX, math.Pow(X[i], 5))
		Xdata = append(Xdata, sliceX)

		//setup T matrix
		var sliceT []float64
		sliceT = append(sliceT, T[i])
		//add to T matrix
		Tdata = append(Tdata, sliceT)
	}

	klog.Infof("Xdata: =%v\n", Xdata)
	klog.Infof("Tdata: =%v\n", Tdata)

	trainCount := int(math.Round(float64(len(Xdata)) * 0.5))
	testCount := len(Xdata) - trainCount
	trainIndexs := UniqueRandomSlice(0, len(Xdata), trainCount)

	klog.Infof("Training count: =%v\n", trainCount)
	klog.Infof("Testing count: =%v\n", testCount)
	klog.Infof("Train indexes: =%v\n", trainIndexs)

	//create matrices for the training and test data.
	var Xtrain [][]float64
	var Ttrain [][]float64
	var Xtest [][]float64
	var Ttest [][]float64
	for i := 0; i < len(Xdata); i++ {
		isTrainIndex := false
		for _, v := range trainIndexs {
			if v == i {
				isTrainIndex = true
				break
			}
		}
		if isTrainIndex {
			Xtrain = append(Xtrain, Xdata[i])
			Ttrain = append(Ttrain, Tdata[i])
		} else {
			Xtest = append(Xtest, Xdata[i])
			Ttest = append(Ttest, Tdata[i])
		}
	}

	klog.Infof("Xtrain: =%v\n", Xtrain)
	klog.Infof("Ttrain: =%v\n", Ttrain)
	klog.Infof("Xtest: =%v\n", Xtest)
	klog.Infof("Ttest: =%v\n", Ttest)

	fmt.Println("train")
	for i := 0; i < len(Xtrain); i++ {
		fmt.Printf("{x: %v, y: %v},", Xtrain[i][1], Ttrain[i][0])
	}
	fmt.Println()
	fmt.Println("test")
	for i := 0; i < len(Xtest); i++ {
		fmt.Printf("{x: %v, y: %v},", Xtest[i][1], Ttest[i][0])
	}
	fmt.Println()

	modelCount := 100
	var models []model
	for i := 0; i < modelCount; i++ {
		var Xselected [][]float64
		var Tselected [][]float64
		//random sample of rows, dup's ok.
		trainIndex := 0
		for j := 0; j < len(Ttrain); j++ {
			trainIndex = rand.Intn(len(Ttrain))

			Xselected = append(Xselected, Xtrain[trainIndex])
			Tselected = append(Tselected, Ttrain[trainIndex])
		}

		model := Train(Xselected, Tselected)

		models = append(models, model)
	}

	var predicted [][]float64
	for _, model := range(models) {
		var predictRow []float64
		prediction := Use(model, Xtest)
		for _, y := range(prediction) {
			predictRow = append(predictRow, y[0])
		}
		predicted = append(predicted, predictRow)
	}

	yMean := MeanByColumn(predicted)
	klog.Infof("yMean: =%v\n", yMean)

	var yTest [][]float64
	for i := 0; i < len(yMean); i++ {
		var ySlice []float64
		ySlice = append(ySlice, yMean[i])
		yTest = append(yTest, ySlice)
	}

	var rmse []float64
	for c := 0; c < len(Ttest[0]); c++ {
		rmse = append(rmse, 0.0)
	}
	//square each of the error values in the matrix
	testError := matrix.Subtract(yTest, Ttest)
	for r := 0; r < len(testError); r++ {
		for c := 0; c < len(testError[r]); c++ {
			testError[r][c] = testError[r][c] * testError[r][c]
		}
	}
	//find avg per column
	errorMeans := MeanByColumn(testError)
	for c := 0; c < len(rmse); c++ {
		rmse[c] = math.Sqrt(errorMeans[c])
	}
	klog.Infof("rmse= %v\n", rmse)

	//for plotting
	fmt.Println("test")
	for i := 0; i < len(Xtest); i++ {
		fmt.Printf("{x: %v, y: %v},", Xtest[i][1], yTest[i][0])
	}
	fmt.Println()

	var xPlot [][]float64
	for i := 0; i < 125; i++ {
		var xPlotSlice []float64
		xVal := 0.10 * float64(i)
		xPlotSlice = append(xPlotSlice, 1.0) //constant
		xPlotSlice = append(xPlotSlice, xVal)
		xPlotSlice = append(xPlotSlice, math.Pow(xVal, 2))
		xPlotSlice = append(xPlotSlice, math.Pow(xVal, 3))
		xPlotSlice = append(xPlotSlice, math.Pow(xVal, 4))
		xPlotSlice = append(xPlotSlice, math.Pow(xVal, 5))
		xPlot = append(xPlot, xPlotSlice)
	}

	klog.Infof("Xtrain= %v\n", Xtrain)
	klog.Infof("Ttrain= %v\n", Ttrain)
	klog.Infof("xPlot= %v\n", xPlot)

	//for _, model := range(models) {
		model := models[0]
		prediction := Use(model, xPlot)
		//klog.Infof("prediction= %v\n", prediction)
		for i, y := range(prediction) {
			fmt.Printf("{x: %v, y: %v},", xPlot[i][1], y[0])
		}
	//}

		fmt.Println("model 50")
	  model = models[50]
		prediction = Use(model, xPlot)
		//klog.Infof("prediction= %v\n", prediction)
		for i, y := range(prediction) {
			fmt.Printf("{x: %v, y: %v},", xPlot[i][1], y[0])
		}
}