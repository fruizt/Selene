/*
Part of the study of Machine Learning

Perceptron. A neural model

Categorization

Given and input return the category
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "perceptron.h"

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

static void calculate_weighted_sum(double x_0[], double x_1[], double z[])
{
    for (int i = 0; i < size; i++)
    {
        z[i] = (w_0 * x_0[i]) + (w_1 * x_1[i]) + b;
    }

    return;
}

// Activation function: Step function
static void calculate_step_function(double z[], double infered_y[])
{
    for (int i = 0; i < size; i++)
    {
        if (z[i] >= 0)
        {
            infered_y[i] = 1;
        }
        else
        {
            infered_y[i] = 0;
        }
    }

    return;
}

// Adjusting the parameters based on Rosenblatt's learning rule
static void adjust_parameters(double x_0[], double x_1[], double y[], double infered_y[])
{
    // double current_w_0 = w_0;
    // double current_w_1 = w_1;
    // double current_b = b;

    int count = 0;

    for (int i = 0; i < size; i++)
    {
        double difference_y = (y[i] - infered_y[i]);

        if (difference_y == 0)
        {
            count++;
        }

        w_0 = w_0 + (learning * difference_y * x_0[i]);
        w_1 = w_1 + (learning * difference_y * x_1[i]);
        b = b + (learning * difference_y);
    }

    if (count == size)
    {
        no_corrections = true;
    }
    return;
}

void run_perceptron_model()
{
    /// Dummy test
    int n = 4;

    double x_0[] = {1, 2, 3, 4};
    double x_1[] = {1, 2, 3, 4};

    double y[] = {1, 1, 0, 0};
    // ----------

    size = n;

    int epoch = 0;

    double *z = malloc(size * sizeof(double));
    if (z == NULL)
    {
        fprintf(stderr, "Could not allocate memory for z \n");
        return;
    }

    double *infered_y = malloc(size * sizeof(double));
    if (infered_y == NULL)
    {
        fprintf(stderr, "Could not allocate memory for the infered result \n");
        return;
    }

    // Epoch
    for (; epoch < max_iterations; epoch++)
    {
        calculate_weighted_sum(x_0, x_1, z);

        calculate_step_function(z, infered_y);

        adjust_parameters(x_0, x_1, y, infered_y);

        if (no_corrections == true)
        {
            break;
        }
    }

    free(z);
    free(infered_y);

    printf("epoch: %i \n", epoch);

    printf("w_0: %lf \n", w_0);
    printf("w_1: %lf \n", w_1);
    printf("b: %lf \n", b);
}
