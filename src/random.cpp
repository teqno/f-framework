#include <random>
#include <time.h>

std::default_random_engine initialize_random_seed()
{
    std::default_random_engine generator;
    generator.seed(time(NULL));
    return generator;
}