## Assignment2


class NeuralNet:
    def __init__(self, dataFile, header=True, h=4):
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h represents the number of neurons in the hidden layer
        raw_input = pd.read_csv(dataFile)
        
        # TODO: Remember to implement the preprocess method   
    def preprocessedData (self, raw_input):
        processed_data = self.preprocess(raw_input)
        self.train_dataset, self.test_dataset = train_test_split(processed_data)
        ncols = len(self.train_dataset.columns)
        nrows = len(self.train_dataset.index)
        self.X = self.train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = self.train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        
        input_layer_size = len(self.X[1])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])
        
        self.W_hidden = 2 * np.random.random((input_layer_size, h)) - 1
        self.Wb_hidden = 2 * np.random.random((1, h)) - 1

        self.W_output = 2 * np.random.random((h, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        self.deltaHidden = np.zeros((h, 1))
        self.h = h
        
        def __activation(self, x, activation="sigmoid"):
            if activation == "sigmoid":
                self.__sigmoid(self, x)
        def __activation_derivative(self, x, activation="sigmoid"):
            if activation == "sigmoid":
                self.__sigmoid_derivative(self, x)

        def __sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        
        def __sigmoid_derivative(self, x):
            return x * (1 - x)
            
         """
    ## tanh activation function

    def __activation(self, x, activation="tanh"):
        if activation == "tanh":
            self.__tanh(self, x)
    def _activation_derivative(self, x , activation = "tanh"):
        if activation == "tanh":
            self._tanh_derivative(self,x)
    def _tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    def _tanh_derivative(self,x):
        return 1 - (((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))**2)
   
    ## reLu activation function
            
    def __activation(self, x, activation="reLu"):
        if activation == "reLu":
            self.__reLu(self, x)
    def _activation_derivative(self,x, activation = "reLu"):
        if activation == "reLu":
            self._reLu_derivative(self,x)
    def _reLu(self,x):
        return np.maximum(0,x)
    def _reLu_derivative(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
        """
        
        def train(self, max_iterations=60000, learning_rate=0.25):
            for iteration in range(max_iterations):
                out = self.forward_pass()
                error = 0.5 * np.power((out - self.y), 2)
                self.backward_pass(out, activation="sigmoid")
                ## tanh and reLu
                self.backward_pass(out, activation="tanh")
                self.backward_pass(out, activation="reLu")
                
                update_weight_output = learning_rate * np.dot(self.X_hidden.T, self.deltaOut)
                update_weight_output_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)
                
                update_weight_hidden = learning_rate * np.dot(self.X.T, self.deltaHidden)
                update_weight_hidden_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden)
                
                self.W_output += update_weight_output
                self.Wb_output += update_weight_output_b
                self.W_hidden += update_weight_hidden
                self.Wb_hidden += update_weight_hidden_b
                
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))
                
        def forward_pass(self, activation="sigmoid"):
            in_hidden = np.dot(self.X, self.W_hidden) + self.Wb_hidden
        if activation == "sigmoid":
            self.X_hidden = self.__sigmoid(in_hidden)
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
            
         """
        if activation == "tanh":
            self.X_hidden = self.__tanh(in_hidden)
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        if activation == "tanh":
            out = self.__tanh(in_output)
        if activation == "reLu":
            self.X_hidden = self.__reLu(in_hidden)
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        if activation == "reLu":
            out = self.__reLu(in_output)  
        """
            return out
        def backward_pass(self, out, activation):
        # pass our inputs through our neural network
            self.compute_output_delta(out, activation)
            self.compute_hidden_delta(activation) 

        def compute_output_delta(self, out, activation="sigmoid"):
            if activation == "sigmoid":
                delta_output = (self.y - out) * (self.__sigmoid_derivative(out))

        self.deltaOut = delta_output

        def compute_hidden_delta(self, activation="sigmoid"):
            if activation == "sigmoid":
                delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden))

                self.deltaHidden = delta_hidden_layer
                
                """
    ## tanh
    def compute_output_delta(self, out, activation="tanh"):
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation="tanh"):
        if activation == "tanh":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden))

        self.deltaHidden = delta_hidden_layer
    ## reLu 
    def compute_output_delta(self, out, activation="reLu"):
        if activation == "reLu":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation="reLu"):
        if activation == "reLu":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden))

        self.deltaHidden = delta_hidden_layer
    """
     # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function   
    def predict(self, header = True):
        self.test_dataset.train()
        self.test_dataset.forward_pass()
        self.test_dataset.backward_pass()
        self.test_dataset.compute_output_delta()
        self.test_dataset.compute_hidden_delta()
        # TODO: obtain prediction on self.test_dataset 
        return 0     
        if __name__ == "__main__":
            preprocessedData()
            neural_network = NeuralNet("Autism-Adult-Data.csv")
            neural_network.train()
            testError = neural_network.predict()
            print("Test error = " + str(testError))
