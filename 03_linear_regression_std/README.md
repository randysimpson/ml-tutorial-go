# Linear Regression with Standardization
If you have not checkout out module [02 - Linear Regression on Actual Data](https://github.com/randysimpson/ml-tutorial-go/blob/master/02_linear_regression_applied/README.md), you might want to do so before continuing.

In module [02 - Linear Regression on Actual Data](https://github.com/randysimpson/ml-tutorial-go/blob/master/02_linear_regression_applied/README.md) we looked at the [Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) data set.  At the end of module 2 we saw that the data did not match up to the predictions very well but why?  If we think about how we come to the answer for the target value, we are using 10 input values.  It's almost certain that some of those values have a higher importance than others.  Also some of those values have a much higher range than others.  Lets take a look.

The highest value for the 1st X column (not counting the bias) can be found by using the command `awk -F ',' '$1 > max { max = $1 } END {print max}' winequality-red.csv` and the lowest you can use `awk -F ',' 'NR == 1 || $1 < min { min = $1 } END {print min}' winequality-red.csv`.

```sh
~/ml-tutorial-go/03_linear_regression_std$ awk -F ',' '$1 > max { max = $1 } END {print max}' winequality-red.csv
15.9
~/ml-tutorial-go/03_linear_regression_std$ awk -F ',' 'NR == 1 || $1 < min { min = $1 } END {print min}' winequality-red.csv
4.6
```

After finding the data for each of the columns I've put it into a chart:

```sh
column                     low      high
1 - fixed acidity          4.6      15.9
2 - volatile acidity       0.12     1.58
3 - citric acid            0        1
4 - residual sugar         0.9      15.5
5 - chlorides              0.012    0.611
6 - free sulfur dioxide    1        72
7 - total sulfur dioxide   6        289
8 - density                0.99007  1.00369
9 - pH                     2.74     4.01
10 - sulphates             0.33     2.0
11 - alcohol               8.4      14.9
```

It's easy to see that the difference between 6 - 289 is a lot larger number than between 0.012 to 0.611.  How can we compensate for this, we can standardize the data.

First we import the data, create X and T, then create the training and test data just as we did in [02 - Linear Regression on Actual Data](https://github.com/randysimpson/ml-tutorial-go/blob/master/02_linear_regression_applied/README.md)  We need to get the standardize variables from the training set.  I've created a method to get the mean values of each column from a slice of slices (matrix) and to get the standard deviation.

```go
	//get standardized info about the training data only
	xMeans := MeanByColumn(Xtrain)
	klog.Infof("xMeans= %v\n", xMeans)
	xStds := StdDevByColumn(Xtrain)
	klog.Infof("xStds= %v\n", xStds)
```

The output looks like this:

```sh
I0821 18:27:09.762042       1 main.go:187] xMeans= [1 8.314620797498067 0.5274042220484757 0.26839718530101553 2.5252541047693517 0.08566379984362747 15.996481626270525 46.12744331508991 0.996726551993747 3.31354182955434 0.6540734949179048 10.430427417253034]
I0821 18:27:09.763039       1 main.go:189] xStds= [0 1.7172528817276773 0.17940792210881734 0.19298449023524725 1.3928877458046613 0.04016396645835077 10.521645813435832 32.32721830873545 0.0018900656905347726 0.1540809356283118 0.15952303098047022 1.073255496391243]
```

Now we need to apply this standardization to the X training data model and the X test data model.  For this example I'm going to create new slice of slices based off the standardized data.  This is done by taking the existing value and subtracting the mean for that column, then dividing that value by the standard deviation for that column.

```go
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

And then the output will look like the following.

```sh
I0821 18:27:09.763640       1 main.go:203] XStdTrain= [[1 -0.532606937062114 0.9620298586750183 -1.3907707555868378 -0.4488905201819742 -0.2406087021720985 -0.4748764323438986 -0.3751465158328466 0.5679421681630838 1.2750323045777323 -0.5897173238228014 -0.9600951690606603] ... [1 -1.406095062172083 0.655465915714682 -0.7689591278559226 -0.3770972257824813 -0.2655066414988077 1.5210090377014456 -0.06580966214822868 -0.6648192176810279 1.6644380396567142 0.3505857727148003 -0.21469949888710826]]
I0821 18:27:09.780697       1 main.go:217] XStdTest= [[1 -0.2996767703661226 1.296463250995386 -1.183500213009866 -0.16171734258400286 0.15775832705524837 -0.0947077713828807 0.24352719153638921 0.14467645628527495 -0.34749159158469106 -0.02553546590024023 -0.5873973339738835] ... [1 -1.3478625204980852 -1.2117871914073686 1.04465811969258 0.7715954846094042 -0.464690156112481 0.19041872433788273 -0.12767703288515225 -0.6542375748840724 0.4962208344197713 0.037151407202266645 0.5306961712864455]]
```

That's not so bad, now lets use a lower value for learning rate and epoch and see what happens.

```sh
I0821 18:27:09.782555       1 main.go:233] learning_rate: =0.001
I0821 18:27:09.782600       1 main.go:234] epoch: =5
I0821 18:27:09.782609       1 main.go:235] Initial w: =[[0] [0] [0] [0] [0] [0] [0] [0] [0] [0] [0] [0]]
I0821 18:27:09.791901       1 main.go:270] RMSE = 3.4481336415072366
I0821 18:27:09.803777       1 main.go:270] RMSE = 1.2239930752523003
I0821 18:27:09.811118       1 main.go:270] RMSE = 0.7241697095400447
I0821 18:27:09.818211       1 main.go:270] RMSE = 0.6594232135485513
I0821 18:27:09.825702       1 main.go:270] RMSE = 0.6520167514527121
I0821 18:27:09.825772       1 main.go:273] Final w = [[5.608193497546576] [0.08176167885287879] [-0.19631878153605775] [-0.0016270093959892863] [0.046901519166847976] [-0.07751031402723589] [0.03737930895031949] [-0.061451179045411204] [-0.10167281478476323] [-0.030751354543315217] [0.19621999849098187] [0.23798697192816354]]
I0821 18:27:09.826128       1 main.go:277] predicted y's= [[5.143951763982442] [4.977818610966575] ... [6.006205153049306]]
I0821 18:27:09.826376       1 main.go:285] find sqrt of = -0.004002690414283838
I0821 18:27:09.826432       1 main.go:287] rmse= NaN
```

An NaN for the rmse on the test data?  I added in an output so we can figure out what's going on here.

```go
	testError := matrix.Subtract(predicted, Ttest)
	square := matrix.Multiply(testError, testError)
	sumError := 0.0
	for _, v := range square {
		sumError += v[0]
	}
	klog.Infof("find sqrt of = %v\n", sumError / float64(len(square)))
	rmse = math.Sqrt(sumError / float64(len(square)))
	klog.Infof("rmse= %v\n", rmse)
```

Seems like the computer is having an issue taking the square root of `-0.004002690414283838`.  Must be a really small number.  Let's visualize this data and see what's going on here.  Again, we are going to plot the predicted values on the x axis and the actual/target values on the y axis.

![Image of predicted vs actual](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/03_linear_regression_std/predicted_vs_actual.PNG)

It looks like the predictions around 5 - 6 are pretty good, but the error happens more when there are low qualities or high qualities.  There might not be as many samples with low or high's and that might make it hard to train this data set.  In general the data moves up and to the right along the 45 degree angle.

Let's take a look at the points if we use int instead of float for our outputs.

![Image of predicted vs actual int output](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/03_linear_regression_std/predicted_vs_actual_int.PNG)
