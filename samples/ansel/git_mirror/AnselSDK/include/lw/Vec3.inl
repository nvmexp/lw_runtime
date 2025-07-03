#pragma once
#include <math.h>
#include "Vec3.h"

namespace lw
{
    inline Vec3 operator-(const Vec3& a)
    {
        Vec3 ret = { -a.x, -a.y, -a.z };
        return ret;
    }

    inline Vec3& operator-=(Vec3& lhs, const Vec3& rhs)
    {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
        return lhs;
    }

    inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs)
    {
        Vec3 result = lhs;
        result.x -= rhs.x;
        result.y -= rhs.y;
        result.z -= rhs.z;
        return result;
    }

    inline const Vec3 operator+(const Vec3& a, const Vec3& b)
    {
        Vec3 r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        r.z = a.z + b.z;
        return r;
    }

    inline const Vec3 operator*(float scale, const Vec3& a)
    {
        Vec3 r;
        r.x = scale*a.x;
        r.y = scale*a.y;
        r.z = scale*a.z;
        return r;
    }

    inline Vec3 vecAdd(const Vec3& a, const Vec3& b)
    {
        Vec3 r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        r.z = a.z + b.z;
        return r;
    }

    inline Vec3 vecScale(float scalar, const Vec3& vec)
    {
        Vec3 r;
        r.x = scalar*vec.x;
        r.y = scalar*vec.y;
        r.z = scalar*vec.z;
        return r;
    }

    inline Vec3 vecCross(const Vec3& a, const Vec3& b)
    {
        Vec3 r;
        r.x = a.y*b.z - a.z*b.y;
        r.y = a.z*b.x - a.x*b.z;
        r.z = a.x*b.y - a.y*b.x;
        return r;
    }

    inline void vecNormalize(Vec3& a)
    {
        float len = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
        if (len > 1e-7f)
        {
            float len_recip = 1.0f / len;
            a.x *= len_recip;
            a.y *= len_recip;
            a.z *= len_recip;
        }
    }

    inline float vecDot(const Vec3& a, const Vec3& b)
    {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    inline float vecLength(const Vec3& a)
    {
        return sqrtf(vecDot(a, a));
    }
}
