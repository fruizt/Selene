/*
Part of the study of Machine Learning

Perceptron. A neural model

Categorization

Given and input return the category
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "perceptron_rosenblat.h"

/// Params
static double w_0 = 0;
static double w_1 = 0;
static double b = 0;

static double learning = 0.005;

/// General setting
static int size = 0;

/// Stop criteria
static int max_iterations = 1000;
static bool no_corrections = false;

// Activation function: Step function
static int calculate_step_function(double z)
{
    if (z >= 0)
    {
        return 1;
    }
    return 0;
}

// Adjusting the parameters based on Rosenblatt's learning rule
static void adjust_parameters(double x_0, double x_1, double y, double infered_y)
{
    double difference_y = (y - infered_y);

    w_0 = w_0 + (learning * difference_y * x_0);
    w_1 = w_1 + (learning * difference_y * x_1);
    b = b + (learning * difference_y);

    return;
}

static void run_perceptron(double x_0[], double x_1[], double y[])
{
    double last_w_0 = w_0;
    double last_w_1 = w_1;
    double last_b = b;

    for (int i = 0; i < size; i++)
    {
        double z = (w_0 * x_0[i]) + (w_1 * x_1[i]) + b;

        double infered_y = calculate_step_function(z);

        adjust_parameters(x_0[i], x_1[i], y[i], infered_y);
    }

    if (last_w_0 == w_0 && last_w_1 == w_1 && last_b == b)
    {
        no_corrections = true;
    }

    return;
}

void run_perceptron_rosenblat_model()
{
    /// Dummy test
    int n = 4;

    double x_0[] = {1, 2, 3, 4};
    double x_1[] = {1, 2, 3, 4};

    double y[] = {1, 1, 0, 0};
    // ----------

    size = n;

    int epoch = 0;

    // Epoch
    for (; epoch < max_iterations; epoch++)
    {
        run_perceptron(x_0, x_1, y);

        if (no_corrections == true)
        {
            break;
        }
    }

    printf("epoch: %i \n", epoch);

    printf("w_0: %lf \n", w_0);
    printf("w_1: %lf \n", w_1);
    printf("b: %lf \n", b);
}
