#include <math.h>
#include <stdbool.h>
#include <stdio.h>

/// Model Parameters.
static double slope = 1;
static double b = 0;

/// Stop parameters
// number of max iterations the model runs
static int max_iterations = 100;

// If the change of loss |last loss - current_loss|<= \tolerance is less than tolerance then we stop
static double tolerance = 0.05;
static double tolerance_for_relative_check = 0.00005;

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

        printf("Infered value: %f \n", inference_result[i]);
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

        printf("partial error: (%f - %f)^2  = %f \n", inference_result[i], y[i], partial_result);

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

static void adjust_parameters(double inference_result[], double y[], double x[], int n)
{
    // Gradient
    double delta_m = calcuate_partial_derivate_of_slope(inference_result, y, x, n);
    double delta_b = calcuate_partial_derivate_of_b(inference_result, y, n);

    printf("delta_m: %f \n", delta_m);
    printf("delta_b: %f \n", delta_b);

    // Adjusting parameters
    slope -= (delta_m * alpha);
    b -= (delta_b * alpha);
}

static bool is_absolute_stopping_criteria_satisfied(double current_loss, double prior_loss)
{
    double delta_error = fabs(current_loss - prior_loss);

    if (delta_error <= tolerance)
    {
        printf(">> Absolute error change stopping createria fullfilled: Error diference: %f, tolerance (absolute): %f \n", delta_error, tolerance);
        return true;
    }

    return false;
}

static bool is_relative_stopping_criteria_satisfied(double current_loss, double prior_loss)
{
    double delta_error = fabs(current_loss - prior_loss) / fabs(prior_loss);

    if (delta_error <= tolerance_for_relative_check)
    {
        printf(">> Relative error change stopping createria fullfilled: Error diference: %f, tolerance (relative): %f \n", delta_error, tolerance_for_relative_check);
        return true;
    }

    return false;
}

void run_model()
{
    /// Dummy test
    int n = 4;

    double x[] = {1, 2, 3, 4};
    double y[] = {2, 5, 5, 8};

    double inference_result[n];

    double last_loss = 0;
    double loss;

    for (int i = 0; i < max_iterations; i++)
    {
        printf("ITERATION: %i -----------------------------\n", i);

        calculate_inference(x, inference_result, n);

        loss = calculate_mean_squared_error(inference_result, y, n);

        if (is_relative_stopping_criteria_satisfied(loss, last_loss))
        {
            break;
        }

        printf("Loss: %f \n", loss);

        last_loss = loss;

        adjust_parameters(inference_result, y, x, n);

        printf("\n");
    }

    printf("slope: %f \n", slope);
    printf("b: %f \n", b);

    return;
}
