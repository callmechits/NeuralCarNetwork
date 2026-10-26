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

def backward(p, out, y, cache):
    X, z1, h = cache
    N = X.shape[0]
    dz2 = (out-y)/N
    g = {}
    g["W2"] = h.T @ dz2
    g["b2"] = dz2.sum(axis=0)
    dh = dz2 @ p["W2"].T
    dz1 = dh * (1 - np.tanh(z1) ** 2)
    g["W1"] = X.T @ dz1
    g["b1"] = dz1.sum(axis = 0)
    return g

def grad_check(p, X, y, eps = 1e-5):
    out, cache = forward(p, X)
    g = backward(p, out, y, cache)
    worst = 0.0
    for name, P in p.items():
        it = np.nditer(P, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = P[idx]
            P[idx] = old + eps; Lp = loss_fn(forward(p, X)[0], y)
            P[idx] = old - eps; Lm = loss_fn(forward(p, X)[0], y)
            P[idx] = old
            num = (Lp - Lm) / (2 * eps)
            rel = abs(num - g[name][idx]) / max(1e-12, abs(num) + abs(g[name][idx]))
            worst = max(worst, rel)
    return worst

print(f"Gradient check, worst relative error: {grad_check(params, X, y):.2e} (want < 1e-6)")


lr = 1.0
for step in range(3001):
    out, cache = forward(params, X)
    L = loss_fn(out, y)
    g = backward(params, out, y, cache)
    for k in params:
        params[k] -= lr * g[k]
    if step % 500 == 0
        print(f"step {step:5d}    loss {L:.4f}")

out, _ = forward(params, X)
print("\nInput  -> P(y=1)  prediction")
for xi, pi in zip(X, out[:, 0]):
    print(f"{xi.astype(int)}  ->  {pi:.3f}     {int(pi > 0.5)}")

  
