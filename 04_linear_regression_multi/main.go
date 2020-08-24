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

func main() {
	klog.Infoln("Initializing ml tutorial application");

	data, err :=ReadCSV("winequality-red.csv", 0, 11)
	if err != nil {
		klog.Errorf("Error loading file: %v\n", err)
	}
	klog.Infof("data= %v\n", data)

	sampleSize := len(data)
	//build the X and T slice of slices from the data
	var X [][]float64
	var T [][]float64
	//get the length of the data slice for 1 row
	dataSliceLength := len(data[0])
	for i := 0; i < sampleSize; i++ {
		//setup X matrix
		var sliceX []float64
		//we will add 1 as the first item
		sliceX = append(sliceX, 1.0)
		//we skip the last 2 columns in the data as they will be used in target matrix
		for j := 0; j < dataSliceLength - 2; j++ {
			sliceX = append(sliceX, data[i][j])
		}
		//add to X matrix
		X = append(X, sliceX)

		//setup T matrix
		var sliceT []float64
		sliceT = append(sliceT, data[i][10])
		sliceT = append(sliceT, data[i][11])
		//add to T matrix
		T = append(T, sliceT)
	}

	klog.Infof("X: =%v\n", X)
	klog.Infof("T: =%v\n", T)

	trainCount := int(math.Round(float64(sampleSize) * 0.8))
	testCount := sampleSize - trainCount
	trainIndexs := UniqueRandomSlice(0, sampleSize, trainCount)

	klog.Infof("Training count: =%v\n", trainCount)
	klog.Infof("Testing count: =%v\n", testCount)
	klog.Infof("Train indexes: =%v\n", trainIndexs)

	//create matrices for the training and test data.
	var Xtrain [][]float64
	var Ttrain [][]float64
	var Xtest [][]float64
	var Ttest [][]float64
	for i := 0; i < sampleSize; i++ {
		isTrainIndex := false
		for _, v := range trainIndexs {
			if v == i {
				isTrainIndex = true
				break
			}
		}
		if isTrainIndex {
			Xtrain = append(Xtrain, X[i])
			Ttrain = append(Ttrain, T[i])
		} else {
			Xtest = append(Xtest, X[i])
			Ttest = append(Ttest, T[i])
		}
	}

	klog.Infof("Xtrain: =%v\n", Xtrain)
	klog.Infof("Ttrain: =%v\n", Ttrain)
	klog.Infof("Xtest: =%v\n", Xtest)
	klog.Infof("Ttest: =%v\n", Ttest)

	//get standardized info about the training data only
	xMeans := MeanByColumn(Xtrain)
	klog.Infof("xMeans= %v\n", xMeans)
	xStds := StdDevByColumn(Xtrain)
	klog.Infof("xStds= %v\n", xStds)

	//standardize data, except for the 1st column...
	var XStdTrain [][]float64
	for r := 0; r < len(Xtrain); r++ {
		var rowData []float64
		//1st column is still 1
		rowData = append(rowData, 1.0)
		for c := 1; c < len(Xtrain[0]); c++ {
			val := (Xtrain[r][c] - xMeans[c]) / xStds[c]
			rowData = append(rowData, val)
		}
		XStdTrain = append(XStdTrain, rowData)
	}
	klog.Infof("XStdTrain= %v\n", XStdTrain)

	//standardize the test data.
	var XStdTest [][]float64
	for r := 0; r < len(Xtest); r++ {
		var rowData []float64
		//1st column is still 1
		rowData = append(rowData, 1.0)
		for c := 1; c < len(Xtest[0]); c++ {
			val := (Xtest[r][c] - xMeans[c]) / xStds[c]
			rowData = append(rowData, val)
		}
		XStdTest = append(XStdTest, rowData)
	}
	klog.Infof("XStdTest= %v\n", XStdTest)


	learning_rate := 0.001
	epoch := 5

	//setup weight matrix as initially all zeros.
	var w [][]float64
	for i := 0; i < len(XStdTrain[0]); i++ { //length of row in X
		var sliceFloat []float64
		for j := 0; j < len(Ttrain[0]); j++ { //length of row in T
			sliceFloat = append(sliceFloat, 0.0)
		}
		w = append(w, sliceFloat)
	}
	
	klog.Infof("learning_rate: =%v\n", learning_rate)
	klog.Infof("epoch: =%v\n", epoch)
	klog.Infof("Initial w: =%v\n", w)

	var rmse []float64
	for c := 0; c < len(Ttrain[0]); c++ {
		rmse = append(rmse, 0.0)
	}
	for i := 0; i < epoch; i++ {
		var sqerrorSum []float64
		for c := 0; c < len(Ttrain[0]); c++ {
			sqerrorSum = append(sqerrorSum, 0.0)
		}

		for j := 0; j < trainCount; j++ {
			//get the x values as a slice of slices, currently it's just a single slice.
			var xMatrix [][]float64
			xMatrix = append(xMatrix, XStdTrain[j])

			//multiply it by weight matrix to get predicted value for y using model
			y := matrix.Multiply(xMatrix, w)
			
			//put the target values into slice of slices, currently this is single slice as well.
			var tMatrix [][]float64
			tMatrix = append(tMatrix, Ttrain[j])

			//find the error
			err := matrix.Subtract(tMatrix, y)

			//Find the amount of change to apply to the weight matrix
			xT := matrix.Transpose(xMatrix)
			diff := matrix.Multiply(xT, err)
			change := matrix.MultiplyScaler(diff, learning_rate)

			//add the change to the weight matrix
			w = matrix.Add(w, change)

			//add to sqerrorSum
			for c := 0; c < len(err[0]); c++ {
				sqerrorSum[c] += err[0][c] * err[0][c]
			}
		}
		for c := 0; c < len(Ttrain[0]); c++ {
			rmse[c] = math.Sqrt(sqerrorSum[c] / float64(trainCount))
		}
		klog.Infof("RMSE = %v\n", rmse)
	}

	klog.Infof("Final w = %v\n", w)

	//run the model against the Xtest values
	predicted := matrix.Multiply(XStdTest, w)
	klog.Infof("predicted y's= %v\n", predicted)

	//square each of the error values in the matrix
	testError := matrix.Subtract(predicted, Ttest)
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
}