#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>

// Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

inline float degrees_to_radians(float degrees)
{
    return degrees * pi / 180.0;
}

inline float random_float()
{
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline float random_float(float min, float max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

inline float clamp(float x, float min, float max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}
// Common Headers

#include "ray.h"
#include "vec3.h"
