import numpy as np

#initialising biases as 0 is not recommended ,may lead to a dead networks
X =[[1,2,3,2.5],
    [2,5,-1,2],
    [-1.5,2.7,2.2,-0.8]]

Y = np.array([1,0,0])

class Layer:
    def __init__(self,n_input,n_neuron) -> None:
        self.weights = 0.1 * np.random.randn(n_input, n_neuron)
        self.biases = np.zeros((1,n_neuron))
    def forward(self,input) -> None:
        self.output = np.dot(input,self.weights) + self.biases

class ReLU:
    #no negative values
    def forward(self,input):
        self.output = np.maximum(0,input)

class Softmax:
    #input-> exponent -> normalise ->output
    def forward(self,input):
        exp_val = np.exp(input - np.max(input,axis = 1, keepdims = True))
        prob_dist = exp_val / np.sum(exp_val, axis = 1, keepdims = True)
        self.output = prob_dist

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_crossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:  # y_true is a vector of labels
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # y_true is one-hot encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

layer1 = Layer(4,3)
layer2 = Layer(3,2)

activation1 = ReLU()
activation2 = Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)

loss_func = Loss_crossentropy()
loss = loss_func.calculate(activation2.output,Y)
print(loss)



# input =[[1,2,3,2.5],
#         [2,5,-1,2],
#         [-1.5,2.7,2.2,-0.8]]

# weights =[[0.2,0.8,-0.5,1.0],
#           [0.5,-0.91,0.26,-0.5],
#           [-0.26,-0.27,0.17,0.87]]

# bias = [2,3,0.5]
# output = []

# #vectorised implementation
# for neuron_weights ,neuron_bias in zip(weights,bias):
#     output.append(np.dot(neuron_weights,np.array(input).T) + neuron_bias)
# print(output)


# #non vectorised implemetation 
# output =[] 
# for neuron_weights ,neuron_bias in zip(weights,bias):
#     neuron_output = 0
#     for n_input,weight in zip(input,neuron_weights):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     output.append(neuron_output)
# print(output)
