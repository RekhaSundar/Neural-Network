# Neural-Network
Python 3

A program that implements a simple sigmoid neural network using the delta rule. Use the following activation function:

<img width="309" alt="Formula" src="https://user-images.githubusercontent.com/56769691/116667277-a360b500-a9b9-11eb-9ad3-992630c4022f.png">

where w is the vector of weights including the bias (w0). Treat all attributes and weights as double-precision values. Given are the two data sets named Gauss3 and Gauss4 as csv files. Program reads both data sets and treats the last value of each line as the class (1 being positive and 0 being negative). Implementation of the back propagation algorithm with single sample correction (means you update the weights every data point) is done. For that purpose, a fixed architecture is used for ease and is as given in the Figure below.

<img width="240" alt="Network" src="https://user-images.githubusercontent.com/56769691/116667418-cc814580-a9b9-11eb-82aa-67796a71905c.png">

The learning rate is η = 0.2 and 2 full iterations (meaning two times going through all data points) is done. The initialization is hardcoded.  Machine learning libraries are not used and the program accepts the following parameters:

1. data - The location of the data file (e.g. /media/data/car.csv).

2. eta - The learning rate for the backpropagation.

3. iterations - The number of iterations to calculate.

The output format :

delta_h1 represent the error of h1 and h1 the output of h1 itself. The same is true for other nodes. The first line indicates the initialization for the weights. The rest of the lines represent the weights after updating and the output and error for that iteration along with the original data point.



