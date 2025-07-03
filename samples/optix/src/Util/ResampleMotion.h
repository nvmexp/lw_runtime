
#pragma once

#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <vector>

// Resample aabbs.
// Input:
//  * A series of bounding boxes and times sorted in increasing t.
//  * requested new times t0, ..., t1, sorted in increasing t.  There must be at least 2.
// Output:
//  * New series of bounding boxes that can be piecewise linearly interpolated:
//  at every t in [t0, t1], the new interpolated result bounds the old interpolated result.
//


void resample_motion_aabbs( const std::vector<float>& input_times,
                            const optix::Aabb*        input_boxes,
                            const std::vector<float>& output_times,
                            optix::Aabb*              output_boxes );

// Uniform version
void resample_motion_aabbs( float              input_t0,
                            float              input_t1,
                            unsigned           input_steps,
                            const optix::Aabb* input_boxes,
                            float              output_t0,
                            float              output_t1,
                            unsigned           output_steps,
                            optix::Aabb*       output_boxes );
