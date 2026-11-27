#include <math.h>
#include <stdbool.h>
#include <stdio.h>

/// Model Parameters.
static double slope = 1;
static double b = 0;

/// Stop parameters
// number of max iterations the model runs
static int max_iterations = 1000000;

// If the change of loss |last loss - current_loss|<= \tolerance is less than tolerance then we stop
static double tolerance = 0.05;
static double tolerance_for_relative_check = 0.00000000000005;
static double tolerance_for_gradient_norm = 0.000001;

/// Learning rate
static double alpha = 0.05;

// Optional function that overwrites the initial set of parameters
static void set_parameters_init(double l_slope, double l_b)
{
    slope = l_slope;
    b = l_b;

    return;
}

// y* = mx + b
static double infer(double x)
{
    return (slope * x) + b;
}

static void calculate_inference(double x[], double inference_result[], int n)
{
    for (int i = 0; i < n; i++)
    {
        inference_result[i] = infer(x[i]);

        printf("Infered value: %lf \n", inference_result[i]);
    }

    return;
}

/// Loss function
//  L(m, b) = average over all points of (m·xᵢ + b − yᵢ)²
static double calculate_mean_squared_error(double inference_result[], double y[], int n)
{
    double result = 0;

    for (int i = 0; i < n; i++)
    {
        double partial_result = pow(inference_result[i] - y[i], 2);

        printf("partial error: (%lf - %lf)^2  = %lf \n", inference_result[i], y[i], partial_result);

        result += partial_result;
    }

    result = result * ((double)1 / n);

    return result;
}

static double calcuate_partial_derivate_of_slope(double inference_result[], double y[], double x[], int n)
{
    double result = 0;

    for (int i = 0; i < n; i++)
    {
        double partial_result = (inference_result[i] - y[i]) * x[i];

        result += partial_result;
    }

    result = result * ((double)2 / n);

    return result;
}

static double calcuate_partial_derivate_of_b(double inference_result[], double y[], int n)
{
    double result = 0;

    for (int i = 0; i < n; i++)
    {
        double partial_result = (inference_result[i] - y[i]);

        result += partial_result;
    }

    result = result * ((double)2 / n);

    return result;
}

static void calculate_gradient(double inference_result[], double y[], double x[], int n, double *grad_m, double *grad_b)
{
    // Gradient
    *grad_m = calcuate_partial_derivate_of_slope(inference_result, y, x, n);
    *grad_b = calcuate_partial_derivate_of_b(inference_result, y, n);

    printf("grad_m: %lf \n", *grad_m);
    printf("grad_b: %lf \n", *grad_b);

    return;
}

static void adjust_parameters(double grad_m, double grad_b)
{
    // Adjusting parameters
    slope -= (grad_m * alpha);
    b -= (grad_b * alpha);

    return;
}

static bool is_absolute_stopping_criteria_satisfied(double current_loss, double prior_loss)
{
    double delta_error = fabs(current_loss - prior_loss);

    if (delta_error <= tolerance)
    {
        printf(">> Absolute error change stopping createria fullfilled: Error diference: %lf, tolerance (absolute): %lf \n", delta_error, tolerance);
        return true;
    }

    return false;
}

static bool is_relative_stopping_criteria_satisfied(double current_loss, double prior_loss)
{
    if (prior_loss == 0)
    {
        return is_absolute_stopping_criteria_satisfied(current_loss, prior_loss);
    }

    double delta_error = fabs(current_loss - prior_loss) / fabs(prior_loss);

    if (delta_error <= tolerance_for_relative_check)
    {
        printf(">> Relative error change stopping createria fullfilled: Error diference: %lf, tolerance (relative): %lf \n", delta_error, tolerance_for_relative_check);
        return true;
    }

    return false;
}

static bool is_gradient_norm_stopping_criteria_satisfied(double grad_m, double grad_b)
{
    double result = sqrt(pow(grad_m, 2) + pow(grad_b, 2));

    if (result <= tolerance_for_gradient_norm)
    {
        printf(">> Gradient norm tolerance stopping createria fullfilled: Gradient norm: %lf, tolerance (gradient norm): %lf \n", result, tolerance_for_gradient_norm);
        return true;
    }

    return false;
}

void run_model(double x[], double y[], int size)
{
    int n = size;

    double inference_result[n];

    double last_loss = 0;
    double loss;

    double grad_m;
    double grad_b;

    for (int i = 0; i < max_iterations; i++)
    {
        printf("ITERATION: %i -----------------------------\n", i);

        calculate_inference(x, inference_result, n);

        loss = calculate_mean_squared_error(inference_result, y, n);

        if (i > 0 && is_relative_stopping_criteria_satisfied(loss, last_loss))
        {
            break;
        }

        printf("Loss: %lf \n", loss);

        last_loss = loss;

        calculate_gradient(inference_result, y, x, n, &grad_m, &grad_b);

        if (is_gradient_norm_stopping_criteria_satisfied(grad_m, grad_b))
        {
            break;
        }

        adjust_parameters(grad_m, grad_b);

        printf("\n");
    }

    printf("slope: %lf \n", slope);
    printf("b: %lf \n", b);

    return;
}
