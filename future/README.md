# Linear Regression on Actual Data
If you have not checkout out module [01 - Linear Regression with SGD](https://github.com/randysimpson/ml-tutorial-go/blob/master/01_linear_regression_sgd/README.md), you might want to do so before continuing.

Let's take a look at some real data, how about [Computer Hardware](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware).  Hey I know this is older data, but we can still build a model around it and see how this stuff works with real life data.

First step is to download the data.  Let's get both the `.data` and `.names` files.  If you havn't done this before or it's been a while, just issue the following curl commands:

```sh
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data
```
```sh
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.names
```

The output should look like:

```sh
~/ml-tutorial-go/02_linear_regression_applied$ curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  8726  100  8726    0     0  13785      0 --:--:-- --:--:-- --:--:-- 13785
~/ml-tutorial-go/02_linear_regression_applied$ curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.names
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  2903  100  2903    0     0  14024      0 --:--:-- --:--:-- --:--:-- 14024
```

Let's take a look at the first few lines of the `.data` file and see what we have with the command `head -5 machine.data`.

```sh
~/ml-tutorial-go/02_linear_regression_applied$ head -5 machine.data
adviser,32/60,125,256,6000,256,16,128,198,199
amdahl,470v/7,29,8000,32000,32,8,32,269,253
amdahl,470v/7a,29,8000,32000,32,8,32,220,253
amdahl,470v/7b,29,8000,32000,32,8,32,172,253
amdahl,470v/7c,29,8000,16000,32,8,16,132,132
```

That works it looks like comma seperated file, and there is no heading.  Now read the `.names` file we can see the mapping for each column of the sample data.  The mapping is:

```sh
   1. vendor name: 30 
      (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, 
       dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson, 
       microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, 
       sratus, wang)
   2. Model Name: many unique symbols
   3. MYCT: machine cycle time in nanoseconds (integer)
   4. MMIN: minimum main memory in kilobytes (integer)
   5. MMAX: maximum main memory in kilobytes (integer)
   6. CACH: cache memory in kilobytes (integer)
   7. CHMIN: minimum channels in units (integer)
   8. CHMAX: maximum channels in units (integer)
   9. PRP: published relative performance (integer)
  10. ERP: estimated relative performance from the original article (integer)
```

Lets get the data into our app so that we may consume it with out linear regression model.

```sh
Code block
```

The output from the running application:

```sh
output
```

Now let's plot the data, we will plot MYCT as the Y axis and plot each of the other data items as a seperate chart on the X axis.

Chart Image

We are going to fit a regression line to this data!  Let's try to predict MYCT based off the other values.  We know the sample size based off the amount of rows in the matrix.  Also we need to setup the matrices `X` and `T`.  `T` will represent the target values, and `X` is the input values.

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
```

The output will show:

```sh
output
```

Awesome! We are going to create a training data matrix based on 80% of the X data/T data.  This will be used to train the model.  We can use the other 20% of the data to check our error as test data.



change column formatting from spaces to comma's:

`awk '{print $1","$2","$3","$4","$5","$6","$7","$8}' auto-mpg.data > auto-mpg-csv.data`

remove all rows that include a `?`:

`sed -i '/?/d' auto-mpg-csv.data`



## Wine data

```sh
$ curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 84199  100 84199    0     0   222k      0 --:--:-- --:--:-- --:--:--  222k
```

```sh
$ head -5 winequality-red.csv
"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5
7.8;0.76;0.04;2.3;0.092;15;54;0.997;3.26;0.65;9.8;5
11.2;0.28;0.56;1.9;0.075;17;60;0.998;3.16;0.58;9.8;6
```

```sh
$ sed -i 's/;/,/g' winequality-red.csv
rsimpson@k8s:~/ml-tutorial-go/02_linear_regression_applied$ head -5 winequality-red.csv
"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,5
7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8,5
11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8,6
```

need to remove the 1st row.

```sh
$ tail -n +2 winequality-red.csv > winequality-red.csv.tmp && mv winequality-red.csv.tmp winequality-red.csv
rsimpson@k8s:~/ml-tutorial-go/02_linear_regression_applied$ head -5 winequality-red.csv
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,5
7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8,5
11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8,6
7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,5
```

