# Linear Regression with SGD
One of the most basic machine learning algorithms is to use a linear model to fit the data.  There is a lot of math behind the scenes but a good example is to think of the data points on a 2 dimensional plot and you need to fit a line through the points to be able to estimate.  If you imagine each point along the x axis has a spring attached vertically to the line, and they are all pulling or pushing at the same time, the line will settle in the best position.

We want to think of the data in matrices, so hopefully you remember your matrix multiplication!  Put the `X` data into a matrix and the output data will be in a `T` matrix.  Then to solve the equation of getting from `X` to `T` you need a weights matrix we will use `w`.

![Image of matrix X, T, w](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/01_linear_regression_sgd/matrix.PNG)

We know that `X` x `w` = `T`.  And we know `X` and `T` for a bunch of values but we don't know `w`?  Well that's what we are looking for and with a bunch of math we can actually solve for `w`.  I'll skip the math for now and show you the solution, and it involves the transpose of matrix `X`.

![Image of solving for w](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/01_linear_regression_sgd/solve_for_w.PNG)

This is awesome if we are using a small enough data set, but what if we need to use a huge data set?  Well there is an answer for that, use an incremental way of solving for `w`.  There is some math using derivatives and gradients, but this is really intented to show the algorithm instead of how the math is derived.  The solution is to use the stochastic gradient decent algorithm or SGD.

First we find the predicted values for Y at each sample.  Then we will find the error.  Then we adjust the weights matrix to reflect the errors based on a learning rate which is sometimes called Rho.  We do all these functions over a specific number of epoch's.  Lets see it in action!

## Example 1

Let's walk through a simple example to fit a liner function to a bunch of points.  Using linear regression we will not get the exact function to match all of the target values, we are just trying to get a function that will approximate the values.  So for a non linear problem it will not be as good in some areas.

First, generate some data based on a non-linear function, and we will add a little noise (error) to the data just to simulate real life data and descrepencies.  For a sample function we can use:

f(x) = 2.0 - 0.1x + 0.05 * (x - 7)^2 + noise

Also lets do a sample size of 100.  If we generate random X values between 0 and 10.

```go
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
```

Which will give us output something like the following, I have used `...` to keep the output shorter.  Also it's important to note that these were generated using random numbers so your output should be a little different.

```sh
I0820 16:24:39.693171       1 main.go:32] X: =[6.046602879796196 6.645600532184904 4.246374970712657 0.6563701921747622 ... 8.279280961505588 0.5792625966433577 4.11540363220476]
I0820 16:24:39.693216       1 main.go:33] T: =[1.534838924265527 1.3854913146395889 2.0231673503113314 3.962096863192246 ... 1.3239286800875558 4.103283129500848 2.0156719107108794]
```

Here is a chart to visualize the sample data:

![Image of sample data for function](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/01_linear_regression_sgd/sample_data.PNG)

This is great, but we need matrices for our next steps and currently we just have slices.  To convert it into a shape for a matrix lets make these slices of slices.  Not sure about that, well currently the data is singular X for singular T, so it may sound strange, but in the future we will use data that has multiple inputs (X's) or even multiple outputs (T's).  Here is the code I'm using to convert to slice of slices:

```go
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
```

Now we need to have a learning rate, or sometimes it's called rho.  It will be used to train the weights for the model.  In the future we will play with this value to make sure that we don't overtrain the model.  For now let's use a learning rate of `0.01`.

We also need to descide on the number of iterations to train the model.  We call this epochs, and for this simple data set lets use `10`.

Lets setup the initial weight matrix, and for this problem we will just use a value of `0`.  It's going to be slice of slices that represent a specific size.  The size of the weight matrix needs to be the length of the inputs (X) plus 1, by the size of the output (T).  In our case we have 1 `X` input and 1 `T` ouput, so it will be 2 x 1 matrix.  Let's take a quick look at some code:

```go
	learning_rate := 0.01
	epoch := 10

	//setup weight matrix as initially all zeros.
	var w [][]float64
	for i := 0; i < len(X1[0]); i++ { //length of row in X
		var arrayFloat []float64
		for j := 0; j < len(T[0]); j++ { //length of row in T
			arrayFloat = append(arrayFloat, 0.0)
		}
		w = append(w, arrayFloat)
	}
	
	klog.Infof("learning_rate: =%v\n", learning_rate)
	klog.Infof("epoch: =%v\n", epoch)
	klog.Infof("Initial w: =%v\n", w)
```

If we run the app to this point we have additional output showing:

```sh
I0820 16:24:39.693282       1 main.go:66] learning_rate: =0.01
I0820 16:24:39.693300       1 main.go:67] epoch: =10
I0820 16:24:39.693306       1 main.go:68] Initial w: =[[0] [0]]
```

For the epochs we will use a `for` loop and iterate through each of the `X` inputs with a `T` Target output.  We are going to use some matrix math to get it all done.  First we create a slice of slices (matrix) for out X we are focusing on, and we then multiply it by the weights matrix `w` to get a predicted output.

Then we make the target values corresponding to the X values we are using into a slice of slices (matrix) and we find the error between the prediction and target values `err`.

Now we need to update our weight matrix because of the error.  This is how we get our linear regression model trained.  We will find a value to change the weight matrix based on our learning rate and the error.

```go
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
```

If we run the app to here we can see that the final weight matrix will look like:

```sh
I0820 16:24:39.694468       1 main.go:96] Final w = [[3.562749539621017] [-0.29241599082718706]]
```

And wow, you might think what good does that tell us.  This is the result of our trained model.  So we can now use this weight matrix to find the solution for any X.  We could add more training data and train the model further or we can keep it how it is.  If we save this weight matrix, we could then use it in the future if needed without re-training the model.  Also we could run the app over a number of times and average the weight matrix.  For this simple example we are going to use the computed weight matrix to find our values of Y with the X's we used for the inital data.  Since the data is in a matrix format, we can do all the math at once instead of iterating through each of the sample items.  Check out the code to run the prediction:

```go
	//run the model against the X values
	predicted := matrix.Multiply(X1, w)
	klog.Infof("predicted y's= %v\n", predicted)
```

The output will look something like:

```sh
I0820 16:24:39.694488       1 main.go:100] predicted y's= [[1.7946261673868897] [1.6194696753604867] [2.321041595136308] [3.370816399526803] ... [1.1417553939256946] [3.39336389347442] [2.3593397088560577]]
```

Each of those Y values correspond to the X value from X1.  If we plot the data, we can easily see the linear regression line that was generated from this model:

![Image of linear regression model for function](https://raw.githubusercontent.com/randysimpson/ml-tutorial-go/master/01_linear_regression_sgd/linear_regression.PNG)

For the sample data set that we used, you can see sometimes the actual values are above the line, and sometimes the values are below the line.  This is because we are using a line to fit the shape of data from a non-linear function.  But in some cases this type of model may work good enough for predicting data, or understanding trends.  In a later module we will learn how to make the model train for better predictions on non-linear functions.

### Example 1 Output

```
~/ml-tutorial-go/01_linear_regression_sgd$ docker run randysimpson/ml-tutorial-go:v1.0
I0820 16:24:39.693052       1 main.go:11] Initializing ml tutorial application
I0820 16:24:39.693171       1 main.go:32] X: =[6.046602879796196 6.645600532184904 4.246374970712657 0.6563701921747622 0.9696951891448456 5.152126285020654 2.1426387258237494 3.1805817433032986 2.830341511804452 6.790846759202163 2.0318687664732287 5.706732760710226 2.9311424455385806 7.525730355516119 8.65335013001561 5.238203060500009 1.5832827774512763 9.752416188605784 5.948085976830626 6.92024587353112 1.7326623818270528 5.44155573000885 4.231522015718281 2.535405005150605 7.8860491501934495 8.80543122741617 8.943617293304536 9.769168685862624 2.2228941700678773 2.4151508854715265 9.32846428518434 8.010550426526613 1.8292491645390843 8.969919575618727 9.789293555766875 0.9083727535388708 9.269868035744143 3.479539636282229 7.109071952999951 6.494894605929405 7.558235074915977 1.3065111702897216 8.963417453962162 7.2114776519267405 0.8552050754191123 6.227283173637045 2.368225468054852 1.8724610140105304 6.2809817121836335 2.8133029380535923 4.349124738914576 5.501469205077233 7.291807267342981 0.005138155161213613 3.9998376285699546 6.039781022829275 0.2967128127488647 0.028430411748625643 5.898341850049194 8.154051709333606 4.584424785756506 0.26265150609689436 2.4969320116349376 2.4746660783662855 5.926237532124455 6.938381365172095 5.39210105890946 7.507630564795985 7.531612777367585 3.557672654092366 2.318300419376769 4.983943012759756 0.2519395979489504 5.893830864007992 5.72086801443084 4.117626883450162 4.916073961316204 7.972085409108028 7.830349733960022 1.304138461737918 7.398257810158336 0.9838378898573259 0.9972966371993512 0.7619026230375504 1.5965092146489503 3.2261068286779753 5.708516273454957 6.841751300974551 5.244997595498651 7.163683749016712 0.12825909106361078 0.98030874806305 8.26454125634742 3.4431501772636057 2.1647114665497034 4.020708452718306 1.6867966833433607 8.279280961505588 0.5792625966433577 4.11540363220476]
I0820 16:24:39.693216       1 main.go:33] T: =[1.534838924265527 1.3854913146395889 2.0231673503113314 3.962096863192246 3.751350472735185 1.7368832309225226 3.00349977374095 2.4582286011393157 2.6155786299622954 1.3449578832458722 3.0670166627155795 1.5992028752847756 2.564374101726659 1.2819048509753133 1.3410152361942342 1.6342064250841646 3.3694383396622225 1.4114934872417684 1.4664296230553437 1.338445716781985 3.2680860265091605 1.6051326163141384 2.013129887435871 2.7712980223831454 1.2868297878566133 1.312147199168833 1.3042661416510806 1.4139269918468282 2.9868554197510426 2.8406792260335263 1.412425763849699 1.3230287133044756 3.1967440018477604 1.3653025479597447 1.502299797327616 3.813873049907981 1.4261227854519123 2.340811978147738 1.3460655988280925 1.4184436172642976 1.3001381410312947 3.5887361086068386 1.3286170565820348 1.3455423529214814 3.869362255485818 1.4440955316874995 2.8893724079967398 3.1512207715215745 1.4104264866296157 2.6361236050558032 1.9789540114303312 1.6244936893041628 1.3581302392288916 4.519497656052496 2.0998517612245124 1.4830847497115114 4.217222063914799 4.518878216478005 1.5267875938698334 1.3389877723226704 1.8933242617400574 4.3279113645300376 2.828366293308373 2.7938313317501233 1.5464639937551659 1.309383961073994 1.6876243258457897 1.2915220143156576 1.276065734014226 2.31990669770692 2.9268689667227124 1.7138135964400716 4.290843618006977 1.564758585015169 1.5685797648965691 2.0588990897868693 1.8213253819573476 1.2607770725013085 1.2907641605612603 3.5107313636127935 1.3335088240749002 3.763364566361308 3.717077059577166 3.9010234904515473 3.314015118065743 2.4434102522637007 1.5638230592621458 1.3823810229532432 1.6949289258828253 1.3486356649867552 4.351283466449849 3.7507144345424424 1.2882672746746504 2.31354399771342 3.008029826994305 2.0923877672810183 3.275963631902918 1.3239286800875558 4.103283129500848 2.0156719107108794]
I0820 16:24:39.693282       1 main.go:66] learning_rate: =0.01
I0820 16:24:39.693300       1 main.go:67] epoch: =10
I0820 16:24:39.693306       1 main.go:68] Initial w: =[[0] [0]]
I0820 16:24:39.694468       1 main.go:96] Final w = [[3.562749539621017] [-0.29241599082718706]]
I0820 16:24:39.694488       1 main.go:100] predicted y's= [[1.7946261673868897] [1.6194696753604867] [2.321041595136308] [3.370816399526803] [3.2791951600868705] [2.056185427119908] [2.9362077136245635] [2.632696577746121] [2.7351124220673997] [1.5769973559733244] [2.9685986210419335] [1.8940096250119676] [2.7056366171532287] [1.3621056410145322] [1.0323715873779342] [2.031015201530903] [3.0997723374929813] [0.7109870968707575] [1.8234340851807926] [1.539158985744662] [3.0560913524700655] [1.971551629189122] [2.3253848366877] [2.8213565728916934] [1.2567426636553027] [0.9879006425954637] [0.9474928272202061] [0.7060883987865689] [2.912739738376641] [2.8565208004487026] [0.8349574127728108] [1.2203364995770913] [3.0278478327025162] [0.9398016192762855] [0.7002035650132492] [3.297126820854528] [0.852091893111623] [2.545276509255079] [1.4839432206227703] [1.6635384981100174] [1.3526007412846632] [3.18070478123396] [0.9417029436229689] [1.4539981567047429] [3.312673900131898] [1.7417923602404703] [2.8702425428775786] [3.0152119969238482] [1.7260900488853979] [2.7400947734930394] [2.29099591986028] [1.9540319710131009] [1.430508492620036] [3.561247060888527] [2.393133056314867] [1.7966209874511532] [3.475985968489936] [3.554436032599918] [1.8379800633014183] [1.1783744297801122] [2.2221904235213135] [3.4859460392234407] [2.8326066914106653] [2.8391176063491104] [1.8298229197875806] [1.5338558779873281] [1.9860129658396828] [1.3673983092517252] [1.3603855268003722] [2.5224291654358093] [2.8848414254538755] [2.105364905318637] [3.4890783724581715] [1.839299147754264] [1.8898762507896607] [2.3586895946402757] [2.1252109012430047] [1.231584285757732] [1.2730300636416967] [3.18139859915608] [1.3993806516685918] [3.275059608245058] [3.271124055305747] [3.339957029191659] [3.0959047157547097] [2.6193843147987925] [1.8934880973655643] [1.5621120539533475] [2.0290283708470653] [1.4679738581796773] [3.5252445304250566] [3.276091585739601] [1.1460655194140208] [2.555917368969675] [2.9297532912749125] [2.387030093592147] [3.0695032161371554] [1.1417553939256946] [3.39336389347442] [2.3593397088560577]]
```