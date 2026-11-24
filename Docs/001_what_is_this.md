In my desire to build an LLM and understand what it is about I want to first understand what is that we are building.

So I take that we are going to build some sort of program that takes input and is going to predict some output.

What is the input? Text, natural language, but it might not be that way. We can train a model with images I would wager. I remember seeing years back models that were able to clasify clouds.

So whatever we decide to build upon we need to understand that given an input (ie, an image of a cloud) the model will predict the output (ie, the type of cloud it is)

This raises a question? Is it easier to build a model that is train for cloud clasification that for natural text?

Maybe it just needs better training?

---

I just read `the bitter lesson` and it was eye opening. Very interesting.

**Search and learn**

---

So I have decided to start with a more historical approach. Learning the main things as they happend. I like this approach as it is more accurate lorewise and it explain the genesis of the ideas. Why do we do X in 2020, well maybe it is because in 2010 the solution Y that was used got stale and could not solve another problem and so X is an 'evolution' of Y.

---

# learning

So let us start on the most basic thing.

So we need to be able to predict stuff, but as we know now we are building models that 'learn'. In this context 'learn' means that the model will automaticallu adjust parameters in it so it can predict better.

So we are not changing the code. The model is taking data and using that data to automatically adjust some sort of variables to be able to predict less wrong.

```
Tom Mitchell's (1997):

"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
```

---

1. So we beginign by the following exercise:

Given a set of points (which have some noise) give a line that fit the best.

So the idea here is that we are going to build a model that will have some parameters which given the data will change to then fit the best.

## Parameters

b -> intersection with Y.
m -> slope of the line.

Given this parameters we will build a line. Then we check how much error this generated line has. In this case it means that we can calculate how bad or good this line is. How? well we got many points that are noise but there is a line that the error with all the points is minimun.

So the best line would hit all the points so the distance on each point generated and given is 0 for all points. This is of course not possible when given points with noise. So what we want to find is the line that has the less average error over all points. this is the best fit line and so we can do this by calculating the line with some initial parameters then checking the errors if the error is to high we move the paremeters (Learning) then we calculate again the line and the errors if we have less error it could be said that we are "moving" in the right directions. And we keep doing this till we get the best fit line.

So here adjusting the parameters requires a way to do it. How can one know if increasing or decreasing the `slope`, same with `b` How can I know if I am moving correctly here? And the answer is computing the gradient.

The gradient tells us the right direction to move the parameters before moving them. So it is not by guessing that we move the parameters but by using the gradient.

## Model

The model is a function that takes some input and predicts the output. This definition is super general.

How does it work?

The model have a set of parameters. Using this parameters the model will generate an output. That output can be 'verfied' against the real answer, basically the prediction (output) is evaluated to see how good it is. This evaluation is call `Loss`.

```
M(m,b)
```

## Loss

Loss is a function that measures **how wrong a model's prediction are**.

Input: the model's prediction and the true answers. The model's predictions are a function of the parameters therefore the loss function also have them as the paramenters. `therefore here is not appropiate I think`

Otput: a number that represent how wrong is the prediction. **Lower is better**.

Given a set of parameters there will be a result. That result is 'evaluated' (the result of the system vs the right answer) using the loss function.

The loss function has the same parameters as the model.

Here in our case `loss` is the following:

```
L(m,b) = average over all the points (m * x_i + b - y_i)^2
```

where `m`, `b` are the same parameters of our model, the slope and the intersect with the `Y` axis respectively. And $x_i$ and $y_i$ are a data point $(x_i,y_i)$

Trainning is the search for parameters that minimize the loss.

The loss in our case for one point:

$({y*}_i - y_i)^2$

Where ${y*}_i = m * x_i + b$ basically ${y*}_i$ is the output of the model on the point $x_i$.

Then of course the square error between the real point and the predicted point is $d(y_i,{y*}_i) =  ({y*}_i - y_i)^2$

Then the loss function in our case is:

$L(m,b) = \frac{1}{N} * \sum_{i=0}^{N} ({y*}_i - y_i)^2$

$L(m,b) = \frac{1}{N} * \sum_{i=0}^{N} (m * x_i + b - y_i)^2$

It can be seen here $L(m,b)$ is function whose parameters are real numbers and is differentiable, Why? because is a polynomial function. Then we can discuss it's gradient.

Let us first take notice of the following:

The derivate of a function of one variable $f(x)$ defines the slope of the tangent line at each of the points.

$f'(x)$

Then for multy variable functions $f(x,y)$ we can define the partial derivates. The partial derivates is the derivate respect to each variable

$\frac{\partial f}{\partial x}$ is the partial derivate of $x$ only.

Then the vector of all partial derivates is called the Gradient

$\nabla f(x, y) = (\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})$
