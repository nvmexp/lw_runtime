/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include <vector>
#include <algorithm>

template <class T>
class MedianSampler
{
private:
    std::vector<T> m_samples;
public:
    MedianSampler() {}
    ~MedianSampler() {}

    inline void add(T s)
    {
        m_samples.push_back(s);
    }

    inline T median() const
    {
        std::vector<T> v(m_samples);
        std::sort(v.begin(), v.end());
        return v[v.size()/2];
    }

    inline const std::vector<T>& samples() const
    {
        return m_samples;
    }
};
