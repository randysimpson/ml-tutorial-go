# Linear Regression on Actual Data
If you have not checkout out module [01 - Linear Regression with SGD](https://github.com/randysimpson/ml-tutorial-go/blob/master/01_linear_regression_sgd/README.md), you might want to do so before continuing.

Let's take a look at some real data, how about [Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).  Hey I know this is older data, but we can still build a model around it and see how this stuff works with real life data.

First step is to download the data.  For now let's just get the `wineequality-red.csv` file.  If you havn't done this before or it's been a while, just issue the following curl command:

```sh
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

The output should look like:

```sh
~/ml-tutorial-go/02_linear_regression_applied$ curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 84199  100 84199    0     0   100k      0 --:--:-- --:--:-- --:--:--  100k
```

Let's take a look at the first few lines of the file and see what we have using the command `head -5 winequality-red.csv`.

```sh
~/ml-tutorial-go/02_linear_regression_applied$ head -5 winequality-red.csv
"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5
7.8;0.76;0.04;2.3;0.092;15;54;0.997;3.26;0.65;9.8;5
11.2;0.28;0.56;1.9;0.075;17;60;0.998;3.16;0.58;9.8;6
```

This file is not comma seperated, it's actually semi-colon seperated.  To take care of that we will just use sed command `sed -i 's/;/,/g' winequality-red.csv`

```sh
~/ml-tutorial-go/02_linear_regression_applied$ sed -i 's/;/,/g' winequality-red.csv
~/ml-tutorial-go/02_linear_regression_applied$ head -5 winequality-red.csv
"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,5
7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8,5
11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8,6
```

We see that this csv file has the heading's for the columns on it as well.  Let's remove them using a tail command `tail -n +2 winequality-red.csv > winequality-red.csv.tmp && mv winequality-red.csv.tmp winequality-red.csv`

```sh
~/ml-tutorial-go/02_linear_regression_applied$ tail -n +2 winequality-red.csv > winequality-red.csv.tmp && mv winequality-red.csv.tmp winequality-red.csv
~/ml-tutorial-go/02_linear_regression_applied$ head -5 winequality-red.csv
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,5
7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8,5
11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8,6
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
```

Now we are ready to get to the data into the app so that we may consume it with out linear regression model.  I wrote a function called `ReadCSV` to get the data and return a matrix, in which you can specify the columns you would like returned based on a 0 index.  You can check out the ReadCSV function if you want but it just reads the data in line by line parsing the data on the commas.  Here is how we retrieve the data usint the function.

```go
	data, err :=ReadCSV("winequality-red.csv", 0, 11)
	if err != nil {
		klog.Errorf("Error loading file: %v\n", err)
	}
	klog.Infof("data= %v\n", data)
```

The output from the running application, using `...` to prevent a large output:

```sh
I0821 17:24:44.264047       1 main.go:75] Initializing ml tutorial application
I0821 17:24:44.265697       1 main.go:81] data= [[7.4 0.7 0 1.9 0.076 11 34 0.9978 3.51 0.56 9.4 5] [7.8 0.88 0 2.6 0.098 25 67 0.9968 3.2 0.68 9.8 5] ... [6 0.31 0.47 3.6 0.067 18 42 0.99549 3.39 0.66 11 6]]
```

The import shows that data now is slice of slices, to represent a matrix!  We know the last column of data is the quality of wine.  Lets focus on predicting that value based on all the other inputs that we know.  Now lets build out our X (input) and T (output) matrices.  Remember we need to add the bias constant column of 1's to the beginning of the X matrix.

```go
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
		//we skip the last index in the data as that will be used in target matrix
		for j := 0; j < dataSliceLength - 1; j++ {
			sliceX = append(sliceX, data[i][j])
		}
		//add to X matrix
		X = append(X, sliceX)

		//setup T matrix
		var sliceT []float64
		sliceT = append(sliceT, data[i][11])
		//add to T matrix
		T = append(T, sliceT)
	}

	klog.Infof("X: =%v\n", X)
	klog.Infof("T: =%v\n", T)
```

Output from the app:

```sh
I0821 17:24:44.278653       1 main.go:108] X: =[[1 7.4 0.7 0 1.9 0.076 11 34 0.9978 3.51 0.56 9.4] [1 7.8 0.88 0 2.6 0.098 25 67 0.9968 3.2 0.68 9.8] ... [1 6 0.31 0.47 3.6 0.067 18 42 0.99549 3.39 0.66 11]]
I0821 17:24:44.285192       1 main.go:109] T: =[[5] [5] ... [6]]
```

Nice, now the data is in the correct shape, we need to seperate the data into train and test matrices.  If we do this, we can train our model on the train data, and then test it against data that has never been used before, but we know the target values.  This will allow us to find out how well our model is working.  We'll dive more into this subject later, but for now let's use 80% of the data as training and the other as test.

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

Output from the app will show:

```sh
I0821 17:24:44.290255       1 main.go:115] Training count: =1279
I0821 17:24:44.290303       1 main.go:116] Testing count: =320
I0821 17:24:44.290313       1 main.go:117] Train indexes: =[1349 606 ... 404]
I0821 17:24:44.294486       1 main.go:141] Xtrain: =[[1 7.4 0.7 0 1.9 0.076 11 34 0.9978 3.51 0.56 9.4] [1 7.8 0.88 0 2.6 0.098 25 67 0.9968 3.2 0.68 9.8] ... [1 5.9 0.645 0.12 2 0.075 32 44 0.99547 3.57 0.71 10.2]]
I0821 17:24:44.303021       1 main.go:142] Ttrain: =[[5] [5] ... [5]]
I0821 17:24:44.306587       1 main.go:143] Xtest: =[[1 7.8 0.76 0.04 2.3 0.092 15 54 0.997 3.26 0.65 9.8] [1 7.4 0.66 0 1.8 0.075 13 40 0.9978 3.51 0.56 9.4] ... [1 6 0.31 0.47 3.6 0.067 18 42 0.99549 3.39 0.66 11]]
I0821 17:24:44.308336       1 main.go:144] Ttest: =[[5] [5] ... [6]]
```

The learning rate, epoch and weight matrix must be setup like in the 1st module [01 - Linear Regression with SGD](https://github.com/randysimpson/ml-tutorial-go/blob/master/01_linear_regression_sgd/README.md).

```go
	learning_rate := 0.0000001
	epoch := 20

	//setup weight matrix as initially all zeros.
	var w [][]float64
	for i := 0; i < len(Xtrain[0]); i++ { //length of row in X
		var sliceFloat []float64
		for j := 0; j < len(Ttrain[0]); j++ { //length of row in T
			sliceFloat = append(sliceFloat, 0.0)
		}
		w = append(w, sliceFloat)
	}
	
	klog.Infof("learning_rate: =%v\n", learning_rate)
	klog.Infof("epoch: =%v\n", epoch)
	klog.Infof("Initial w: =%v\n", w)
```

And the output will be:

```sh
I0821 17:24:44.308483       1 main.go:159] learning_rate: =1e-07
I0821 17:24:44.308494       1 main.go:160] epoch: =20
I0821 17:24:44.308500       1 main.go:161] Initial w: =[[0] [0] [0] [0] [0] [0] [0] [0] [0] [0] [0] [0]]
```

There is a way to measure the error and it's called the Root Mean Squared Error, we will call it RMSE from now on.  We are going to calculate this on each iteration of the epoch.  Other than the rmse the looping function is the same as it was in [01 - Linear Regression with SGD](https://github.com/randysimpson/ml-tutorial-go/blob/master/01_linear_regression_sgd/README.md)

```go
	rmse := 0.0
	for i := 0; i < epoch; i++ {
		sqerrorSum := 0.0

		for j := 0; j < trainCount; j++ {
			//get the x values as a slice of slices, currently it's just a single slice.
			var xMatrix [][]float64
			xMatrix = append(xMatrix, Xtrain[j])

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
			sqerror := matrix.Multiply(err, err) //same as saying err^2
			sqerrorSum += sqerror[0][0] //it becomes a matrix of size 1 x 1

			rmse = math.Sqrt(sqerrorSum / float64(trainCount))
		}
		klog.Infof("RMSE = %v\n", rmse)
	}

	klog.Infof("Final w = %v\n", w)
```

And the output is:

```sh
I0821 17:24:44.318071       1 main.go:196] RMSE = 4.9706991426235705
I0821 17:24:44.327492       1 main.go:196] RMSE = 3.985912466515707
I0821 17:24:44.335695       1 main.go:196] RMSE = 3.5079920830941465
I0821 17:24:44.341705       1 main.go:196] RMSE = 3.28574161676246
I0821 17:24:44.347585       1 main.go:196] RMSE = 3.1787062752778685
I0821 17:24:44.353156       1 main.go:196] RMSE = 3.1200901742473346
I0821 17:24:44.358229       1 main.go:196] RMSE = 3.0811261648825687
I0821 17:24:44.364250       1 main.go:196] RMSE = 3.050042104186349
I0821 17:24:44.369352       1 main.go:196] RMSE = 3.022149879190437
I0821 17:24:44.374746       1 main.go:196] RMSE = 2.995618458102611
I0821 17:24:44.381471       1 main.go:196] RMSE = 2.9697477665027527
I0821 17:24:44.386734       1 main.go:196] RMSE = 2.9442782607769034
I0821 17:24:44.393319       1 main.go:196] RMSE = 2.9191184351610087
I0821 17:24:44.398549       1 main.go:196] RMSE = 2.8942387411233073
I0821 17:24:44.416142       1 main.go:196] RMSE = 2.86963109307032
I0821 17:24:44.423769       1 main.go:196] RMSE = 2.8452939085419384
I0821 17:24:44.431129       1 main.go:196] RMSE = 2.8212269062776523
I0821 17:24:44.436500       1 main.go:196] RMSE = 2.797429517426492
I0821 17:24:44.443110       1 main.go:196] RMSE = 2.773900558945611
I0821 17:24:44.451010       1 main.go:196] RMSE = 2.750638296684587
I0821 17:24:44.451206       1 main.go:199] Final w = [[0.005064354833516434] [0.04333446947910843] [0.002485571061153467] [0.0013949601848591473] [0.011014653595739863] [0.0004108829827484399] [0.039240678098323545] [0.056124563584433594] [0.005046309091536539] [0.01682029791847243] [0.0033878528513580355] [0.054856940530202083]]
```

Wow, we can see that the rmse seems to be moving in larger differences at the beginning but towards the end it's small changes.

Let's run the model on our test data and find the error of that!

```go
	//run the model against the Xtest values
	predicted := matrix.Multiply(Xtest, w)
	klog.Infof("predicted y's= %v\n", predicted)

	testError := matrix.Subtract(predicted, Ttest)
	square := matrix.Multiply(testError, testError)
	sumError := 0.0
	for _, v := range square {
		sumError += v[0]
	}
	rmse = math.Sqrt(sumError / float64(len(square)))
	klog.Infof("rmse= %v\n", rmse)
```

Output observed:

```sh
I0821 17:24:44.451821       1 main.go:203] predicted y's= [[4.58939162165116] [3.6839753487169014] ... [4.037448177339826]]
I0821 17:24:44.452112       1 main.go:212] rmse= 0.7334594542114395
```

The rmse of 0.73 means that on average of all the test data we are 0.73 off on our quality when using the inputs provided.  That sounds like it's pretty good, less than 1, but again it's averaged across all the test data.  Let's plot out some of the data to visualize what's happening here.  We are going to plot the predicted values on the x axis and the actual/target values on the y axis.

![Image of predicted vs actual](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/02_linear_regression_applied/predicted_vs_actual.PNG)

So looking at this plot the data that is closest to the 45 degree line is when the predictions are accurate, and the data farther from that line is when the predications are bad.  Using a chart like this it is easy to pick out outliers from the data, maybe they have an unusual input or there was bad input data.  Let's get this data into int form to match the output values more closely by rounding and take a look at that chart.

![Image of predicted vs actual int output](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/02_linear_regression_applied/predicted_vs_actual_int.PNG)

Yikes, got to say it's not looking so good here.  Let's move on to the next module and see what we can do!

## Complete Code

[main.go](https://github.com/randysimpson/ml-tutorial-go/blob/master/02_linear_regression_applied/main.go)

## Complete Output

[output.txt](https://github.com/randysimpson/ml-tutorial-go/blob/master/02_linear_regression_applied/output.txt)