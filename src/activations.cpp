#include <math.h>

#include "activations.h"

double sigmoid(double a) {
    return 1 / (1 + exp(a));
}