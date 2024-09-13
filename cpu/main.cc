/* 
    EC527 Final Project
    By Will Martin, Jordan Nichols

    Code adopted from "Ray Tracing in One Weekend", by Peter Shirley
    https://raytracing.github.io/books/RayTracingInOneWeekend.html

*/
#include <string>
#include <fstream>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;
using std::ofstream;
using std::string;

#include "rtweekend.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "Timer.h"
#include <omp.h>

#define NUM_THREADS 16
#define MAX_DEPTH 50
#define SAMPLES_PER_PIXEL 20
#define ASPECT_RATIO (16.0f / 9.0f)
#define IMG_WIDTH 120
#define IMG_HEIGHT static_cast<int>(IMG_WIDTH / ASPECT_RATIO)

typedef void (*ray_function)(camera &, hittable_list &, color[], int, int);

void driver(ray_function func, string name, camera &cam, hittable_list &world, color pixel_colors[], int use_threads)
{
    string bla = use_threads ? "Multi-Threaded " : "Single-Threaded ";
    cerr << "Testing " << bla << name << " Code..." << endl;
#pragma omp parallel for if (use_threads)
    for (int j = IMG_HEIGHT - 1; j >= 0; --j)
    {
        for (int i = 0; i < IMG_WIDTH; ++i)
        {
            func(cam, world, pixel_colors, i, j);
        }
    }
}

color ray_color(const ray &r, const hittable &world, int depth)
{
    hit_record rec;

    /* If we've exceeded the ray bounce limit, no more light is gathered */
    if (depth <= 0)
        return color(0, 0, 0);

    if (world.hit(r, 0.001, infinity, rec))
    {
        /* If we hit an object, calculate scattered rays based on the 
        material of the object */
        ray scattered;
        color attenuation;
        /* If we scatter, then we recurse on the attenuated scattered ray */
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return color(0, 0, 0);
    }
    /* If we hit nothing, return gradient based on y value */
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

/* predefined scene used for benchmarking */
hittable_list set_scene()
{
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto choose_mat = random_float();
            point3 center(a + 0.9, 0.2, b + 0.9);

            if ((center - point3(4, 0.2, 0)).length() > 0.9)
            {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8)
                {
                    // diffuse
                    auto albedo = color::random(0.7, 0.7) * color::random(0.7, 0.7);
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95)
                {
                    // metal
                    auto albedo = color::random(0.5, 0.5);
                    auto fuzz = random_float(0, 0);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else
                {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}

hittable_list random_scene()
{
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto choose_mat = random_float();
            point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

            if ((center - point3(4, 0.2, 0)).length() > 0.9)
            {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8)
                {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95)
                {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_float(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else
                {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5f);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0f, material1));

    auto material2 = make_shared<lambertian>(color(0.4f, 0.2f, 0.1f));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0f, material2));

    auto material3 = make_shared<metal>(color(0.7f, 0.6f, 0.5f), 0.0f);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0f, material3));

    return world;
}

/* No loop unrolling or accumulators */
void ray_trace_unopt(camera &cam, hittable_list &world, color pixel_colors[], int i, int j)
{
    color pixel_color(0, 0, 0);
    for (int s = 0; s < SAMPLES_PER_PIXEL; ++s)
    {
        auto u = (i + random_float()) / (IMG_WIDTH - 1);
        auto v = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, world, MAX_DEPTH);
    }
    pixel_colors[((IMG_HEIGHT - j - 1) * IMG_WIDTH) + i] = color(pixel_color.x(), pixel_color.y(), pixel_color.z());
}

/* Loop unrolling x2 */
void ray_trace_u2(camera &cam, hittable_list &world, color pixel_colors[], int i, int j)
{
    color pixel_color(0, 0, 0);
    for (int s = 0; s < SAMPLES_PER_PIXEL; s += 2)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u2 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v2 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        pixel_color += ray_color(r1, world, MAX_DEPTH);
        pixel_color += ray_color(r2, world, MAX_DEPTH);
    }
    if (SAMPLES_PER_PIXEL % 2)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        pixel_color += ray_color(r1, world, MAX_DEPTH);
    }
    pixel_colors[((IMG_HEIGHT - j - 1) * IMG_WIDTH) + i] = color(pixel_color.x(), pixel_color.y(), pixel_color.z());
}

/* Loop unrolling x4 */
void ray_trace_u4(camera &cam, hittable_list &world, color pixel_colors[], int i, int j)
{
    color pixel_color(0, 0, 0);
    for (int s = 0; s < SAMPLES_PER_PIXEL; s += 4)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u2 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u3 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u4 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v2 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v3 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v4 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        ray r3 = cam.get_ray(u3, v3);
        ray r4 = cam.get_ray(u4, v4);
        pixel_color += ray_color(r1, world, MAX_DEPTH);
        pixel_color += ray_color(r2, world, MAX_DEPTH);
        pixel_color += ray_color(r3, world, MAX_DEPTH);
        pixel_color += ray_color(r4, world, MAX_DEPTH);
    }
    for (int s = 0; s < SAMPLES_PER_PIXEL % 4; s++)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        pixel_color += ray_color(r1, world, MAX_DEPTH);
    }
    pixel_colors[((IMG_HEIGHT - j - 1) * IMG_WIDTH) + i] = color(pixel_color.x(), pixel_color.y(), pixel_color.z());
}

/* Loop unrolling x8 */
void ray_trace_u8(camera &cam, hittable_list &world, color pixel_colors[], int i, int j)
{
    color pixel_color(0, 0, 0);
    int s;
    for (s = 0; s < SAMPLES_PER_PIXEL; s += 8)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u2 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u3 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u4 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u5 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u6 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u7 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u8 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v2 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v3 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v4 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v5 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v6 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v7 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v8 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        ray r3 = cam.get_ray(u3, v3);
        ray r4 = cam.get_ray(u4, v4);
        ray r5 = cam.get_ray(u5, v5);
        ray r6 = cam.get_ray(u6, v6);
        ray r7 = cam.get_ray(u7, v7);
        ray r8 = cam.get_ray(u8, v8);
        pixel_color += ray_color(r1, world, MAX_DEPTH);
        pixel_color += ray_color(r2, world, MAX_DEPTH);
        pixel_color += ray_color(r3, world, MAX_DEPTH);
        pixel_color += ray_color(r4, world, MAX_DEPTH);
        pixel_color += ray_color(r5, world, MAX_DEPTH);
        pixel_color += ray_color(r6, world, MAX_DEPTH);
        pixel_color += ray_color(r7, world, MAX_DEPTH);
        pixel_color += ray_color(r8, world, MAX_DEPTH);
    }
    for (int s = 0; s < SAMPLES_PER_PIXEL % 8; s++)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        pixel_color += ray_color(r1, world, MAX_DEPTH);
    }
    pixel_colors[((IMG_HEIGHT - j - 1) * IMG_WIDTH) + i] = color(pixel_color.x(), pixel_color.y(), pixel_color.z());
}

/* Loop unrolling x2, 2 accumulators */
void ray_trace_u2_a2(camera &cam, hittable_list &world, color pixel_colors[], int i, int j)
{
    color pixel_color1(0, 0, 0);
    color pixel_color2(0, 0, 0);
    for (int s = 0; s < SAMPLES_PER_PIXEL; s += 2)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u2 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v2 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        pixel_color1 += ray_color(r1, world, MAX_DEPTH);
        pixel_color2 += ray_color(r2, world, MAX_DEPTH);
    }
    if (SAMPLES_PER_PIXEL % 2)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        pixel_color1 += ray_color(r1, world, MAX_DEPTH);
    }
    pixel_colors[((IMG_HEIGHT - j - 1) * IMG_WIDTH) + i] = color(pixel_color1.x() + pixel_color2.x(), pixel_color1.y() + pixel_color2.y(), pixel_color1.z() + pixel_color2.z());
}

/* Loop unrolling x4, 2 accumulators */
void ray_trace_u4_a2(camera &cam, hittable_list &world, color pixel_colors[], int i, int j)
{
    color pixel_color1(0, 0, 0);
    color pixel_color2(0, 0, 0);
    for (int s = 0; s < SAMPLES_PER_PIXEL; s += 4)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u2 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u3 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u4 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v2 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v3 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v4 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        ray r3 = cam.get_ray(u3, v3);
        ray r4 = cam.get_ray(u4, v4);
        pixel_color1 += ray_color(r1, world, MAX_DEPTH);
        pixel_color1 += ray_color(r2, world, MAX_DEPTH);
        pixel_color2 += ray_color(r3, world, MAX_DEPTH);
        pixel_color2 += ray_color(r4, world, MAX_DEPTH);
    }
    for (int s = 0; s < SAMPLES_PER_PIXEL % 4; s++)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        pixel_color1 += ray_color(r1, world, MAX_DEPTH);
    }
    pixel_colors[((IMG_HEIGHT - j - 1) * IMG_WIDTH) + i] = color(pixel_color1.x() + pixel_color2.x(), pixel_color1.y() + pixel_color2.y(), pixel_color1.z() + pixel_color2.z());
}

/* Loop unrolling x8, 2 accumulators */
void ray_trace_u8_a2(camera &cam, hittable_list &world, color pixel_colors[], int i, int j)
{
    color pixel_color1(0, 0, 0);
    color pixel_color2(0, 0, 0);
    int s;
    for (s = 0; s < SAMPLES_PER_PIXEL; s += 8)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u2 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u3 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u4 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u5 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u6 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u7 = (i + random_float()) / (IMG_WIDTH - 1);
        auto u8 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v2 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v3 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v4 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v5 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v6 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v7 = (j + random_float()) / (IMG_HEIGHT - 1);
        auto v8 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        ray r2 = cam.get_ray(u2, v2);
        ray r3 = cam.get_ray(u3, v3);
        ray r4 = cam.get_ray(u4, v4);
        ray r5 = cam.get_ray(u5, v5);
        ray r6 = cam.get_ray(u6, v6);
        ray r7 = cam.get_ray(u7, v7);
        ray r8 = cam.get_ray(u8, v8);
        pixel_color1 += ray_color(r1, world, MAX_DEPTH);
        pixel_color1 += ray_color(r2, world, MAX_DEPTH);
        pixel_color1 += ray_color(r3, world, MAX_DEPTH);
        pixel_color1 += ray_color(r4, world, MAX_DEPTH);
        pixel_color2 += ray_color(r5, world, MAX_DEPTH);
        pixel_color2 += ray_color(r6, world, MAX_DEPTH);
        pixel_color2 += ray_color(r7, world, MAX_DEPTH);
        pixel_color2 += ray_color(r8, world, MAX_DEPTH);
    }
    for (int s = 0; s < SAMPLES_PER_PIXEL % 8; s++)
    {
        auto u1 = (i + random_float()) / (IMG_WIDTH - 1);
        auto v1 = (j + random_float()) / (IMG_HEIGHT - 1);
        ray r1 = cam.get_ray(u1, v1);
        pixel_color1 += ray_color(r1, world, MAX_DEPTH);
    }
    pixel_colors[((IMG_HEIGHT - j - 1) * IMG_WIDTH) + i] = color(pixel_color1.x() + pixel_color2.x(), pixel_color1.y() + pixel_color2.y(), pixel_color1.z() + pixel_color2.z());
}

int main()
{

    // Image
    color *pixel_colors = new color[IMG_WIDTH * IMG_HEIGHT];

    // World -- set_scene() is used for testing, change to random_scene() for different image output
    auto world = set_scene();

    // Camera
    point3 lookfrom(0, 5, 15);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    float dist_to_focus = 15.8f;
    float aperture = 0.1f;

    camera cam(lookfrom, lookat, vup, 20, ASPECT_RATIO, aperture, dist_to_focus);

    // Render

    ray_function functions[7] = {ray_trace_unopt, ray_trace_u2, ray_trace_u4, ray_trace_u8, ray_trace_u2_a2, ray_trace_u4_a2, ray_trace_u8_a2};
    string names[7] = {"Unoptimized", "2x Unroll", "4x Unroll", "8x Unroll", "2x Unroll, 2 Accumulators", "4x Unroll, 2 Accumulators", "8x Unroll, 8 Accumulators"};

    cout << "P3\n"
         << IMG_WIDTH << ' ' << IMG_HEIGHT << "\n255\n";

    cerr << "Image Size:\t" << IMG_WIDTH << "x" << IMG_HEIGHT << endl;
    cerr << "Max Depth:\t" << MAX_DEPTH << endl;
    cerr << "Samples/Pixel:\t" << SAMPLES_PER_PIXEL << endl;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

    /* Using OpenMP */
    for (int i = 0; i < 7; i++)
    {
        driver(functions[i], names[i], cam, world, pixel_colors, 1);
    }
    /* Single-Threaded Code */
    for (int i = 0; i < 7; i++)
    {
        driver(functions[i], names[i], cam, world, pixel_colors, 0);
    }
    write_colors(std::cout, pixel_colors, IMG_WIDTH * IMG_HEIGHT, SAMPLES_PER_PIXEL);

    cerr << "\nDone.\n";
}