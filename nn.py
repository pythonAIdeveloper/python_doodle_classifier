import random
import math

def sigmoid(x, a, b):
    return 1/(1 + math.exp(-x))

def dsigmoid(y, a, b):
    return y * (1 - y)

class Matrix():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = []

        for i in range(self.rows):
            self.data.append([])
            for j in range(self.cols):
                self.data[i].append(0)

#vector operations
    @staticmethod
    def MatrixMultiply(m1, m2):
        if m1.cols != m2.rows:
            return "you did something wrong"
        else:
            result = Matrix(m1.rows, m2.cols)
            for i in range(result.rows):
                for j in range(result.cols):
                    sum = 0
                    for k in range(m1.cols):
                        sum += m1.data[i][k] * m2.data[k][j]
                    result.data[i][j] = sum
            return result

#converting array into matrix
    @staticmethod
    def fromArray(arr):
        m = Matrix(len(arr), 1)
        for i in range(len(arr)):
            m.data[i][0] = arr[i]
        return m

#converting matrix into array
    def toArray(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr

#scalar operations
    def multiply(self, n):
        if type(n) == Matrix:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    def add(self, n):
        if type(n) == Matrix:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]

        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n

    @staticmethod
    def subtract(a, b):
        result = Matrix(a.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        return result

# randomization
    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.uniform(-1, 1)

# transposing a matrix
    @staticmethod
    def transpose(m):
        result = Matrix(m.cols, m.rows)
        for i in range(m.rows):
            for j in range(m.cols):
                result.data[j][i] = m.data[i][j]
        return result

# applying a function
    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = func(self.data[i][j], i, j)

    @staticmethod
    def staticMap(m, func):
        result = Matrix(m.rows, m.cols)
        for i in range(m.rows):
            for j in range(m.cols):
                val = m.data[i][j]
                result.data[i][j] = func(val, 0, 0)
        return result

# printing a matrix
    def printMatrix(self):
        b = []
        a = []
        for i in range(self.rows):
            a = []
            for j in range(self.cols):
                a.append([self.data[i][j]])
            b.append(a)
        for i in range (len(b)):
            print(b[i])
        print()

class SingleLayerNewralNetwork():
    def __init__(self, inputNodes, hiddenNodes, OutputNodes):

        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = OutputNodes

        self.weights_ih = Matrix(self.hiddenNodes, self.inputNodes)
        self.weights_ho = Matrix(self.outputNodes, self.hiddenNodes)

        self.bias_h = Matrix(self.hiddenNodes, 1)
        self.bias_o = Matrix(self.outputNodes, 1)

        self.weights_ho.randomize()
        self.weights_ih.randomize()
        self.bias_h.randomize()
        self.bias_o.randomize()

        self.learningRate = 0.1

    def feedForward(self, inputArray):
        # generating hidden outputs
        inputs = Matrix.fromArray(inputArray)

        hidden = Matrix.MatrixMultiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        # activation function
        hidden.map(sigmoid)

        # generating the output's output
        output = Matrix.MatrixMultiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        # activation function
        output.map(sigmoid)

        # done!
        return output.toArray()

    def train(self, inputs_array, targets_array):

        # generating hidden's outputs
        inputs = Matrix.fromArray(inputs_array)
        hidden = Matrix.MatrixMultiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        # activation function
        hidden.map(sigmoid)

        # generating the output's outputs
        outputs = Matrix.MatrixMultiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        # activation function
        outputs.map(sigmoid)

        targets = Matrix.fromArray(targets_array)

        # error = targets - outputs

        # formula->
        # del(W) = lr * E * ($ * ($-1) * H)

        outputErrors = Matrix.subtract(targets, outputs)

        # calculating gradient
        gradients = Matrix.staticMap(outputs, dsigmoid)
        gradients.multiply(outputErrors)
        gradients.multiply(self.learningRate)

        # calculationg deltas
        hiddenT = Matrix.transpose(hidden)
        weight_ho_deltas = Matrix.MatrixMultiply(gradients, hiddenT)

        # adjust the weights and biases
        self.weights_ho.add(weight_ho_deltas)
        self.bias_o.add(gradients)

        # calculating hidden error
        weight_ho_t = Matrix.transpose(self.weights_ho)
        hiddenErrors = Matrix.MatrixMultiply(weight_ho_t, outputErrors)

        # calculating hidden layer gradient
        hiddenGradient = Matrix.staticMap(hidden, dsigmoid)
        hiddenGradient.multiply(hiddenErrors)
        hiddenGradient.multiply(self.learningRate)

        # calculate input -> hidden deltas
        inputsT = Matrix.transpose(inputs)
        weight_ih_deltas = Matrix.MatrixMultiply(hiddenGradient, inputsT)

        # adjust the weights and biases
        self.weights_ih.add(weight_ih_deltas)
        self.bias_h.add(hiddenGradient)

class MultiLayerNewralNetwork():

    def __init__(self, inputNodes, arrayOfHiddenNodes, OutputNodes):

        self.inputNodes = inputNodes
        self.hiddenLayers = arrayOfHiddenNodes
        self.outputNodes = OutputNodes

        self.weights_ih = Matrix(self.hiddenLayers[0], self.inputNodes)
        self.weights_ho = Matrix(self.outputNodes, self.hiddenLayers[len(self.hiddenLayers)-1])
        self.weights_h = []
        for i in range(len(self.hiddenLayers)):
            if i != 0:
                self.weights_h.append(Matrix(self.hiddenLayers[i], self.hiddenLayers[i-1]))

        self.bias_o = Matrix(self.outputNodes, 1)
        self.bias_h = []
        for i in range(len(self.hiddenLayers)):
            self.bias_h.append(Matrix(self.hiddenLayers[i], 1))


        for i in range(len(self.weights_h)):
            self.weights_h[i].randomize()
        self.weights_ho.randomize()
        self.weights_ih.randomize()

        for i in range(len(self.bias_h)):
            self.bias_h[i].randomize()
        self.bias_o.randomize()

        self.learningRate = 0.1

    def feedForward(self, inputArray):
        # generating hidden outputs
        inputs = Matrix.fromArray(inputArray)


        hidden = Matrix.MatrixMultiply(self.weights_ih, inputs)
        hidden.add(self.bias_h[0])
        hidden.map(sigmoid)

        hiddenVals = []
        hiddenVals.append(hidden)
        for i in range(len(self.weights_h)):
            a = Matrix.MatrixMultiply(self.weights_h[i], hiddenVals[len(hiddenVals)-1])
            a.add(self.bias_h[i+1])
            a.map(sigmoid)
            hiddenVals.append(a)

        output = Matrix.MatrixMultiply(self.weights_ho, hiddenVals[len(hiddenVals)-1])
        output.add(self.bias_o)
        output.map(sigmoid)

        # done!
        return output.toArray()

    def train(self, inputs_array, targets_array):

        # generating hidden outputs
        inputs = Matrix.fromArray(inputs_array)

        hidden = Matrix.MatrixMultiply(self.weights_ih, inputs)
        hidden.add(self.bias_h[0])
        hidden.map(sigmoid)

        hiddenVals = []
        hiddenVals.append(hidden)
        for i in range(len(self.weights_h)):
            a = Matrix.MatrixMultiply(self.weights_h[i], hiddenVals[len(hiddenVals) - 1])
            a.add(self.bias_h[i + 1])
            a.map(sigmoid)
            hiddenVals.append(a)

        output = Matrix.MatrixMultiply(self.weights_ho, hiddenVals[len(hiddenVals) - 1])
        output.add(self.bias_o)
        output.map(sigmoid)

        """
        feed forward part over
        """

        targets = Matrix.fromArray(targets_array)
        outputErrors = Matrix.subtract(targets, output)

        # calculating gradient
        gradients = Matrix.staticMap(output, dsigmoid)  # dsigmoid the next layer
        gradients.multiply(outputErrors)  # multiply errors of nest layer
        gradients.multiply(self.learningRate)  # multiply lr

        # calculationg deltas
        hiddenT = Matrix.transpose(hiddenVals[len(hiddenVals) - 1])  # transpose previouse layer
        weight_ho_deltas = Matrix.MatrixMultiply(gradients, hiddenT)  # multiply to gradient

        # adjust the weights and biases
        self.weights_ho.add(weight_ho_deltas)  # add deltas
        self.bias_o.add(gradients)  # add gradients to the next layer

        hiddenErrors = [outputErrors]

        hiddenGradients = []
        hiddenGradients.append(gradients)
        # gradients.printMatrix()
        weightss = []

        for i in self.weights_h:
            weightss.append(i)

        weightss.append(self.weights_ho)

        for i in range(len(self.weights_h)):
            weight  = weightss[len(weightss) - 1 - i]
            weightT = Matrix.transpose(weight)
            error = Matrix.MatrixMultiply(weightT, hiddenErrors[0])
            hiddenErrors.insert(0, error)

            gradient = Matrix.staticMap(hiddenVals[len(hiddenVals) - 1 - i], dsigmoid)
            gradient.multiply(error)
            gradient.multiply(self.learningRate)

            previous_layer_t = Matrix.transpose(hiddenVals[len(hiddenVals) - 2 - i])
            delta = Matrix.MatrixMultiply(gradient, previous_layer_t)
            self.weights_h[len(self.weights_h) - 1 - i].add(delta)
            self.bias_h[len(self.bias_h) - 1 - i].add(gradient)

            hiddenGradients.append(gradient)

        wT = Matrix.transpose(weightss[0])
        e = Matrix.MatrixMultiply(wT, hiddenErrors[0])

        g = Matrix.staticMap(hiddenVals[0], dsigmoid)
        g.multiply(e)
        g.multiply(self.learningRate)

        # calculate input -> hidden deltas
        inputsT = Matrix.transpose(inputs)
        weight_ih_deltas = Matrix.MatrixMultiply(g, inputsT)

        # adjust the weights and biases
        self.weights_ih.add(weight_ih_deltas)
        self.bias_h[0].add(g)

if __name__ == '__main__':
    print("This is a Newral Network Library")
    print("-By CHAITANYA JAIN")