# Perceptron

The idea is that this thing works like a neuron.

A neuron recieves signals, weighs them, and fires if the total is high enough.

A perceptron is an abstraction of such behaviour.

A perceptron is a neuron model

It takes various inputs and calculate the weighted sum of them, adds a bias, and passes the result to an activation function  to produce an output.

```
the classic perceptron has three parts:

1. A linear discriminant: the weighted sum z = w·x + b.
2. A step activation: output 1 if z ≥ 0, else 0.
3. Rosenblatt's learning rule: update the weights only on mistakes, in proportion to the error.
```

## Linear Discriminants

Linear: a line
Discriminant: a separation

A discriminant is a function that assigns an input to a class.

All function can be used as discriminants. "Discriminant" is not a a special kind of function but rather a **role**.

## Exercise

Write the weight sum function.

$z = w_1 * x_1 + w_2 * x_2 + b$

Write the activation function. In this case we will use the `Step` function

```md
  1 |          ●━━━━━━━━━
    |          |
  0 |━━━━━━━━━○
    +----------+---------  z
               0
```

Apply Rosenblatt's learning rule

$w_1 = w_1 + \eta * (y-\hat{y}) * x_1$
$w_2 = w_2 + \eta * (y-\hat{y}) * x_2$
$b = b + \eta * (y-\hat{y})$

Where $\eta$ is the learning rate.

One full pass through all the points is called an epoch.
