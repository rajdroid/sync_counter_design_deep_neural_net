import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime


x_train = np.array([
  #| Current state | input |
    [[0,              0]], 
    [[0,              1]], 
    [[1,              0]], 
    [[1,              1]],
    [[2,              0]],
    [[2,              1]],
    [[3,              0]],
    [[3,              1]]
    ])

# New state
y_train = np.array([
    [[1]], 
    [[3]], 
    [[2]], 
    [[0]],
    [[3]],
    [[1]],
    [[0]],
    [[2]]
    ])


# Network
net = Network()
net.add(FCLayer(2, 4))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(4, 1))

# Train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=4000, learning_rate=0.02)

# Test
out = net.predict(x_train)
x = np.round(out)

# Take user input and perform state transitions
user_input = int(input("Enter the input (0 or 1 or -1):"))
current_state = 0
print("Current state is", current_state)

while user_input != -1:
	user_state_input = np.array([[[current_state, user_input]]])
	new_state = net.predict(user_state_input)
	current_state = np.round(new_state).item()

	print("Current state is", current_state)
	user_input = int(input("Enter the input (0 or 1 or -1):"))