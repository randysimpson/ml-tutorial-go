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

	//tmp
	i := 0

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

		//fmt.Printf("{x: %v, y: %v},", columns[beginColumn + 7], columns[beginColumn])
		result = append(result, data)

		i += 1
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

	data, err :=ReadCSV("machine.data", 2, 9)
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
		//we skip the 0 index in the data as that will be used in target matrix
		for j := 1; j < dataSliceLength; j++ {
			sliceX = append(sliceX, data[i][j])
		}
		//add to X matrix
		X = append(X, sliceX)

		//setup T matrix
		var sliceT []float64
		sliceT = append(sliceT, data[i][0])
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
	epoch := 50

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

	rmse := 0.0
	for i := 0; i < epoch; i++ {
		sqerrorSum := 0.0

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
			change := matrix.MultiplyScalar(diff, learning_rate)

			//add the change to the weight matrix
			w = matrix.Add(w, change)

			//add to sqerrorSum
			sqerror := matrix.Multiply(err, err) //same as saying err^2
			sqerrorSum += sqerror[0][0] //it becomes a matrix of size 1 x 1

			rmse = math.Sqrt(sqerrorSum / float64(trainCount))
		}
		klog.Infof("RMSE = %v\n", rmse)
	}

	klog.Infof("Final w = %v\n", w)

	//run the model against the Xtest values
	predicted := matrix.Multiply(XStdTest, w)
	klog.Infof("predicted y's= %v\n", predicted)

	//prints predicted vs test
	/*for i := 0; i < len(predicted); i++ {
		fmt.Printf("{x: %v, y: %v},", i, predicted[i][0])
	}
	fmt.Println()
	for i := 0; i < len(predicted); i++ {
		fmt.Printf("{x: %v, y: %v},", i, Ttest[i][0])
	}*/

	//plot on chart with predicted on x and actual on y
	for i := 0; i < len(predicted); i++ {
		fmt.Printf("{x: %v, y: %v},", predicted[i][0], Ttest[i][0])
	}

	//run the model against the Xtrain values
	/*predicted := matrix.Multiply(Xtrain, w)
	for i := 0; i < len(predicted); i++ {
		fmt.Printf("{x: %v, y: %v},", predicted[i][0], Ttrain[i][0])
	}*/

	testError := matrix.Subtract(predicted, Ttest)
	square := matrix.Multiply(testError, testError)
	sumError := 0.0
	for _, v := range square {
		sumError += v[0]
	}
	rmse = math.Sqrt(sumError / float64(len(square)))
	klog.Infof("rmse= %v\n", rmse)
}