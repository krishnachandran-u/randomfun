from __future__ import annotations
from typing import Callable
import math 

class Value: 
    def __init__(self, data: float, _children: set[Value] = ()):
        self.data = data 
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def backward(self) -> None:
        topological_ordering: list[Value] = []
        seen = set()
        def topological_sort(parent: Value):
            if parent not in seen:
                seen.add(parent)
                for child in parent._prev:
                    topological_sort(child)
                topological_ordering.append(parent)
                
        topological_sort(self)
        self.grad = 1
        for value in reversed(topological_ordering):
           value._backward() 

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __add__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: Value) -> Value:
        assert isinstance(other, (int, float)), f"Exponent has invalid type: {type(other)}"
        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        out = Value(math.tanh(self.data), (self, ))

        def _backward():
            self.grad += (1 - math.tanh(self.data)**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(self.data, 0), (self, ))

        def _backward():
            self.grad += (1 if self.data >=0 else 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        _sigmoid = lambda x: 1/(1 + math.exp(-x))

        out = Value(_sigmoid(self.data), (self, ))

        def _backward():
            self.grad += (_sigmoid(self.data)*(1 - _sigmoid(self.data))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def _rmul__(self, other):
        return self * other

    def __sub__ (self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1