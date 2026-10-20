import numpy as np
rng = np.random.default_rng(0)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = float)
y = np.array([[0], [1], [1], [0]], dtype = float)

H = 4
params = {
    "W1": rng.normal(0, 0.5, (2, H)),
    "b1": np.zeroes(H),
    "W2": rng.normal(0, 0.5, (H, 1)),
    "b2": np.zeros(1),
}

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]
    h = np.tanh(z1)
    z2 = h @ p["W2"] + p["b2"]
    out = sigmoid(z2)
    return out, (X, z1, h)

def loss_fn(out, y):
    eps = 1e-12
    return -np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))


  
