#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

X = np.array([[0.05, 0.10]])  
Y = np.array([[0.01, 0.99]]) 

input_neurons = 2
hidden_neurons = 2
output_neurons = 2

w_input_hidden = np.array([[0.15, 0.20], [0.25, 0.30]])
w_hidden_output = np.array([[0.40, 0.45], [0.50, 0.55]])

b_hidden = np.array([[0.35, 0.35]])
b_output = np.array([[0.60, 0.60]])

learning_rate = 0.5

epochs = 10000
for epoch in range(epochs):
    
    hidden_input = np.dot(X, w_input_hidden) + b_hidden
    hidden_output = tanh(hidden_input)
    
    output_input = np.dot(hidden_output, w_hidden_output) + b_output
    output = tanh(output_input)
    
    error = Y - output
    mse = np.mean(error ** 2)
    
    output_error_term = error * tanh_derivative(output)
    hidden_error_term = np.dot(output_error_term, w_hidden_output.T) * tanh_derivative(hidden_output)
    
    w_hidden_output += learning_rate * np.dot(hidden_output.T, output_error_term)
    w_input_hidden += learning_rate * np.dot(X.T, hidden_error_term)
    b_output += learning_rate * output_error_term
    b_hidden += learning_rate * hidden_error_term
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.6f}")

print("Final Output after training:", output)
print("Final weights from input to hidden layer:", w_input_hidden)
print("Final weights from hidden to output layer:", w_hidden_output)
print("Final hidden biases:", b_hidden)
print("Final output biases:", b_output)


# In[ ]:




