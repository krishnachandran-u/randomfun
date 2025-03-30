import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, n_in, **kwargs):
        self.w: list[Value] = [Value(random.uniform(1, -1)) for _ in range(n_in)]
        self.b: Value = Value(0)
        self.nonlin: bool = True

    def __call__(self, x: list[float]):
        act: Value = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act 

    def params(self):
        return self.w + [self.b] 

    def __repr__(self):
        return f"Neuron(w={self.w}, b={self.b})"

class Layer:
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons: list[Neuron] = [Neuron(n_in, **kwargs) for _ in range(n_out)] 

    def __call__(self, x: list[float]):
        return [neuron(x) for neuron in self.neurons]
        
    def params(self):
        return [p for neuron in self.neurons for p in neuron.params()] 

    def __repr__(self):
        return f"Layer[{', '.join(str(neuron) for neuron in self.neurons)}]"

class MLP:
    def __init__(self, n_in, n_outs):
        n_all: list[int] = [n_in] + n_outs
        self.layers: list[Layer] = [Layer(n_all[i], n_all[i + 1], nonlin = bool(i != len(n_outs) - 1)) for i in range(len(n_outs))]

    def __call__(self, x: list[float]):
        for layer in self.layers:
            x = layer(x)
        return x 

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def __repr__(self):
        return f"MLP{{{', '.join(str(layer) for layer in self.layers)}}}"