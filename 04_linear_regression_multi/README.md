# Linear Regression with Multiple Outputs
In this module we are going to rely on the data set that was utilized in [03 - Linear Regression with Standardization](https://github.com/randysimpson/ml-tutorial-go/blob/master/03_linear_regression_std/README.md).  In that module we predicted the quality of the wine based on the input variables.  In this module we are going to estimate the quality of the wine as well as the alcohol level.

I've updated the X matrix to be everything but the last 2 columns and the T matrix to be those items.  The code is very similar to [03 - Linear Regression with Standardization](https://github.com/randysimpson/ml-tutorial-go/blob/master/03_linear_regression_std/README.md)

```go
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
```

With the output we can verify what the matrices have the correct shape.

```sh
I0821 22:52:27.366989       1 main.go:148] X: =[[1 7.4 0.7 0 1.9 0.076 11 34 0.9978 3.51 0.56] [1 7.8 0.88 0 2.6 0.098 25 67 0.9968 3.2 0.68] ... [1 6 0.31 0.47 3.6 0.067 18 42 0.99549 3.39 0.66]]
I0821 22:52:27.457304       1 main.go:149] T: =[[9.4 5] [9.8 5] ... [10.2 5] [11 6]]
```

Let's split our data sets up fo a test and training set.  We do this the exact same way we did it on [03 - Linear Regression with Standardization](https://github.com/randysimpson/ml-tutorial-go/blob/master/03_linear_regression_std/README.md).

```go
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
```

The output we can again verify the shapes of our matrices.

```sh
I0821 22:52:27.479066       1 main.go:155] Training count: =1279
I0821 22:52:27.479941       1 main.go:156] Testing count: =320
I0821 22:52:27.481421       1 main.go:157] Train indexes: =[1349 606 ... 404]
I0821 22:52:27.492661       1 main.go:181] Xtrain: =[[1 7.4 0.7 0 1.9 0.076 11 34 0.9978 3.51 0.56] [1 7.8 0.88 0 2.6 0.098 25 67 0.9968 3.2 0.68] ... 0.71]]
I0821 22:52:27.529746       1 main.go:182] Ttrain: =[[9.4 5] [9.8 5] ... [11 6] [10.2 5]]
I0821 22:52:27.543258       1 main.go:183] Xtest: =[[1 7.8 0.76 0.04 2.3 0.092 15 54 0.997 3.26 0.65] [1 7.4 0.66 0 1.8 0.075 13 40 0.9978 3.51 0.56] ... [1 6 0.31 0.47 3.6 0.067 18 42 0.99549 3.39 0.66]]
I0821 22:52:27.551728       1 main.go:184] Ttest: =[[9.8 5] [9.4 5] ... [9.5 6] [11 6]]
```

We need to standardize the data, using the train data for standardization but applying it to both training and test data.

```go
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
```

The output will be similar to the following.

```sh
I0821 22:52:27.553824       1 main.go:188] xMeans= [1 8.314620797498067 0.5274042220484757 0.26839718530101553 2.5252541047693517 0.08566379984362747 15.996481626270525 46.12744331508991 0.996726551993747 3.31354182955434 0.6540734949179048]
I0821 22:52:27.561086       1 main.go:190] xStds= [0 1.7172528817276773 0.17940792210881734 0.19298449023524725 1.3928877458046613 0.04016396645835077 10.521645813435832 32.32721830873545 0.0018900656905347726 0.1540809356283118 0.15952303098047022]
I0821 22:52:27.563989       1 main.go:204] XStdTrain= [[1 -0.532606937062114 0.9620298586750183 -1.3907707555868378 -0.4488905201819742 -0.2406087021720985 -0.4748764323438986 -0.3751465158328466 0.5679421681630838 1.2750323045777323 -0.5897173238228014] ... [1 -1.406095062172083 0.655465915714682 -0.7689591278559226 -0.3770972257824813 -0.2655066414988077 1.5210090377014456 -0.06580966214822868 -0.6648192176810279 1.6644380396567142 0.3505857727148003]]
I0821 22:52:27.587076       1 main.go:218] XStdTest= [[1 -0.2996767703661226 1.296463250995386 -1.183500213009866 -0.16171734258400286 0.15775832705524837 -0.0947077713828807 0.24352719153638921 0.14467645628527495 -0.34749159158469106 -0.02553546590024023] ... [1 -1.3478625204980852 -1.2117871914073686 1.04465811969258 0.7715954846094042 -0.464690156112481 0.19041872433788273 -0.12767703288515225 -0.6542375748840724 0.4962208344197713 0.037151407202266645]]
```

From here we get into the training loop.  Set the variables and then train the model.  We are going to ouput the RMSE at each epoch to visualize the convergence.

```go
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
```

The output shows that the weight matrix has 2 columns, 1 for each output predicted.

```sh
I0821 22:52:27.610691       1 main.go:234] learning_rate: =0.001
I0821 22:52:27.611244       1 main.go:235] epoch: =5
I0821 22:52:27.611506       1 main.go:236] Initial w: =[[0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0]]
I0821 22:52:27.657280       1 main.go:279] RMSE = [6.289586029747064 3.4553715172186683]
I0821 22:52:27.685776       1 main.go:279] RMSE = [1.988528166779666 1.2170112624689373]
I0821 22:52:27.718925       1 main.go:279] RMSE = [0.8902493776915057 0.7423051053972998]
I0821 22:52:27.747907       1 main.go:279] RMSE = [0.7010518281541115 0.6818387807082392]
I0821 22:52:27.772925       1 main.go:279] RMSE = [0.6642745020525664 0.6736373345102824]
I0821 22:52:27.773024       1 main.go:282] Final w = [[10.411256106209349 5.610286525551262] [0.40473705256285775 0.1415598323049567] [-0.014238005839973266 -0.20624145719419618] [0.23425870260840023 0.05005052919986832] [0.28579301870273505 0.10303118421272268] [-0.14544880281767555 -0.11280261134470564] [-0.05695939268064518 0.02802455781031957] [-0.14407271246278358 -0.10034714825313498] [-0.8905677666602213 -0.2773276659408225] [0.36591649396424697 0.03255637871046369] [0.19111910019464037 0.23645343431306917]]
```

That works, not to run the model against the test data.

```go
	//run the model against the Xtest values
	predicted := matrix.Multiply(XStdTest, w)
	klog.Infof("predicted y's= %v\n", predicted)
```

And the output has 2 columns as well 1st one representing the alcohol level and the 2nd representing the quality.

```sh
I0821 22:52:27.773838       1 main.go:286] predicted y's= [[9.63452955575009 5.122222366563467] [9.640764951317212 5.044760123723691] ... [11.194672389912947 6.078132778197388]]
```

Now we want to measure the test data using rmse.

```go
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
```

And the output will be similar to the following.

```sh
I0821 22:52:27.780035       1 main.go:300] rmse= [0.6736940366442123 0.6902610976187404]
```

Excellent, so with all that we can see that the rmse on the test data is very low number.  Let's plot this data in a chart so we can visualize the predicted vs actual.

![Image of predicted vs actual alcohol](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/04_linear_regression_multi/predicted_vs_actual_alcohol.PNG)

![Image of predicted vs actual quality](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/04_linear_regression_multi/predicted_vs_actual_quality.PNG)

Not too shabby, with this simple linear regression algorithm, we could identify outliers and make some predictions for data.
