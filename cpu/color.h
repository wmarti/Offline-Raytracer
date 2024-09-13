#pragma once

#include "vec3.h"

#include <iostream>
//Using our new vec3 class, we'll create a utility function to write a single pixel's color out to the standard output stream.
void write_color(std::ostream &out, color pixel_color, int samples_per_pixel)
{
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    //Divide the color by the number of samples

    auto scale = 1.0f / (float)samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

void write_colors(std::ostream &out, color *pixel_colors, int len, int samples_per_pixel)  {
    for (int i = 0; i < len; i++)
    {
        color pixel_color = pixel_colors[i];

        auto r = pixel_color.x();
        auto g = pixel_color.y();
        auto b = pixel_color.z();

        //Divide the color by the number of samples

        auto scale = 1.0f / (float)samples_per_pixel;
        r = sqrt(scale * r);
        g = sqrt(scale * g);
        b = sqrt(scale * b);

        // Write the translated [0,255] value of each color component.
        out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
    }
}           