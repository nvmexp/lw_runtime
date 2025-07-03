#include <anselutils/internal/Math.h>
#include <lw/Vec3.inl>
#include <math.h>
#include <assert.h>

namespace anselutils
{
namespace internal
{
    using namespace lw;

    float determineChirality(const Vec3& right, const Vec3& up, const Vec3& forward)
    {
        Vec3 lookAtInRightHandedCoordinates = vecCross(up, right);
        float chirality = vecDot(lookAtInRightHandedCoordinates, forward) > 0.0f ? 1.0f : -1.0f;
        return chirality;
    }

    void makeQuaternionFromRotationMatrix(Quat& q, const float r[3][3])
    {
        float trace = r[0][0] + r[1][1] + r[2][2];

        if (trace > 0.0f) {
            // |w| > 1/2, may as well choose w > 1/2
            float s = sqrtf(trace + 1.0f);  // 2w

            q.w = s * 0.5f;
            s = 0.5f / s;       // 1/(4w)

            q.x = (r[2][1] - r[1][2]) * s;
            q.y = (r[0][2] - r[2][0]) * s;
            q.z = (r[1][0] - r[0][1]) * s;
        }
        else {
            // |w| <= 1/2
            int i = 0;
            if (r[1][1] > r[0][0]) i = 1;
            if (r[2][2] > r[i][i]) i = 2;

            int j = (1 << i) & 3; // i + 1 modulo 3.
            int k = (1 << j) & 3;

            float s = sqrtf(r[i][i] - r[j][j] - r[k][k] + 1.0f);

            float* component = &q.x;
            component[i] = s * 0.5f;
            s = 0.5f / s;
            component[j] = (r[i][j] + r[j][i]) * s;
            component[k] = (r[k][i] + r[i][k]) * s;
            q.w = (r[k][j] - r[j][k]) * s;
        }
    }

    void makeRotationMatrixFromEulerAngles(float rot[3][3], float pitch, float yaw, float roll)
    {
        float sp = sinf(pitch);
        float cp = cosf(pitch);

        float sy = sinf(yaw);
        float cy = cosf(yaw);

        float sr = sinf(roll);
        float cr = cosf(roll);

        // The base rotation matrix is
        //
        // [ cos(a) -sin(a) ]
        // [ sin(a)  cos(a) ]
        //
        // And the direction of rotation is from the first component axis to the second (e.g. from x
        // to y). Internally we use a right handed coordinate system with y up. We also use column 
        // vectors internally.
        // 
        // The rotation order is roll, pitch and last yaw. Our combined rotation matrix is 
        // therefore:
        //
        //     [ cos(y)  0  sin(y) ][ 1    0       0    ][ cos(r)  sin(r)  0]
        // R = [   0     1    0    ][ 0  cos(p)  sin(p) ][-sin(r)  cos(r)  0]
        //     [-sin(y)  0  cos(y) ][ 0 -sin(p)  cos(p) ][   0       0     1]
        //
        // Note: We use -y for yaw angle because we want to rotate from x-axis to z-axis for 
        //       positive values of y.
        //       We use -p for pitch angle because we want to rotate from z-axis to y-axis for 
        //       positive values of p.
        //       We use -r for roll angle because we want to rotate from y-axis to x-axis for 
        //       positive values of r.
        //
        rot[0][0] = cr*cy + sr*sp*sy;
        rot[0][1] = cy*sr - sp*sy*cr;
        rot[0][2] = cp*sy;

        rot[1][0] = -sr*cp;
        rot[1][1] = cr*cp;
        rot[1][2] = sp;

        rot[2][0] = sr*sp*cy - cr*sy;
        rot[2][1] = -sr*sy - cr*sp*cy;
        rot[2][2] = cp*cy;
    }

    void makeRotationMatrixFromQuaternion(float rot[3][3], const Quat& q)
    {
        float s = 2.0f;

        float xs = q.x * s;
        float ys = q.y * s;
        float zs = q.z * s;

        float wx = q.w * xs;
        float wy = q.w * ys;
        float wz = q.w * zs;

        float xx = q.x * xs;
        float xy = q.x * ys;
        float xz = q.x * zs;

        float yy = q.y * ys;
        float yz = q.y * zs;
        float zz = q.z * zs;

        rot[0][0] = 1.0f - (yy + zz);
        rot[0][1] = xy - wz;
        rot[0][2] = xz + wy;

        rot[1][0] = xy + wz;
        rot[1][1] = 1.0f - (xx + zz);
        rot[1][2] = yz - wx;

        rot[2][0] = xz - wy;
        rot[2][1] = yz + wx;
        rot[2][2] = 1.0f - (xx + yy);
    }

    void matMult(float res[3][3], const float a[3][3], const float b[3][3])
    {
        // inputs should be different from output container
        assert(a != res && b != res);

        // for now implement this in the straight-forward manner - we need a math library eventually
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                res[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
    }

    void matSetAndTranspose(float dest[3][3], float source[3][3])
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                dest[i][j] = source[j][i];
            }
        }
    }

    void vecMult(Vec3& res, const float m[3][3], const Vec3& vec)
    {
        res.x = m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z;
        res.y = m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z;
        res.z = m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z;
    }

    void calcEulerAnglesFromRotationMatrix(float& roll, float& pitch, float& yaw, float rot[3][3])
    {
        pitch = atan2f(rot[1][2], sqrtf(rot[2][2] * rot[2][2] + rot[0][2] * rot[0][2]));
        yaw = atan2f(rot[0][2], rot[2][2]);
        roll = atan2f(-rot[1][0], rot[1][1]);
        if (fabsf(roll) < 1e-7f)
        {
            roll = 0.0f;
        }
    }

    void makeQuaternionFromRotationAxisAngle(Quat& q, const Vec3& axis, float angle)
    {
        float a_half = 0.5f*angle;
        q.w = cosf(a_half);
        float sin_a_half = sinf(a_half);
        q.x = axis.x*sin_a_half;
        q.y = axis.y*sin_a_half;
        q.z = axis.z*sin_a_half;
    }

    void quatMult(Quat& res, const Quat& q1, const Quat& q2)
    {
        float a1 = q1.w;
        float b1 = q1.x;
        float c1 = q1.y;
        float d1 = q1.z;

        float a2 = q2.w;
        float b2 = q2.x;
        float c2 = q2.y;
        float d2 = q2.z;

        res.w = a1*a2 - b1*b2 - c1*c2 - d1*d2;
        res.x = a1*b2 + b1*a2 + c1*d2 - d1*c2;
        res.y = a1*c2 - b1*d2 + c1*a2 + d1*b2;
        res.z = a1*d2 + b1*c2 - c1*b2 + d1*a2;
    }

    void quatNormalize(Quat& res)
    {
        float magn = sqrtf(res.w*res.w + res.x*res.x + res.y*res.y + res.z*res.z);
    
        if (magn != 0.0f)
        {
            float imagn = 0.0f;
            imagn = 1.0f / magn;
        
            res.w *= imagn;
            res.x *= imagn;
            res.y *= imagn;
            res.z *= imagn;
        }
        else
        {
            res.w = 1.0f;
            res.x = 0.0f;
            res.y = 0.0f;
            res.z = 0.0f;
        }
    }

} // end of namespace internal
} // end of namespace anselutils
