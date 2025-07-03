#pragma once
#include <lw/Vec3.h>
#include <lw/Quat.h>

namespace anselutils
{
namespace internal
{
    // Given basis vector, determines chirality (left/right) of the coordinate system
    // Returns +1.0f in case it's right-handed, -1.0f in case it's left-handed
    float determineChirality(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward);
    // Colwerts rotation matrix 'r' to quaternion 'q'
    void makeQuaternionFromRotationMatrix(lw::Quat& q, const float r[3][3]);
    // Create quaternion from unit axis and right handed rotation angle (in radians)
    void makeQuaternionFromRotationAxisAngle(lw::Quat& q, const lw::Vec3& axis, float angle);
    // Colwerts quaternion 'q' to rotation matrix 'r'
    void makeRotationMatrixFromQuaternion(float rot[3][3], const lw::Quat& q);
    // Given euler angles constructs a rotation matrix 'rot'
    void makeRotationMatrixFromEulerAngles(float rot[3][3], float pitch, float yaw, float roll);
    // Given rotation matrix, callwlate the roll, pitch and yaw angles
    void calcEulerAnglesFromRotationMatrix(float& roll, float& pitch, float& yaw, float rot[3][3]);
    // Sets matrix 'dest' to a transposed version of 'source'
    void matSetAndTranspose(float dest[3][3], float source[3][3]);
    // Multiplies matricex 'a' anb 'b' and puts the result in 'res'
    void matMult(float res[3][3], const float a[3][3], const float b[3][3]);
    // Transforms vector 'vec' by matrix 'm' and puts the result in 'res'
    void vecMult(lw::Vec3& res, const float m[3][3], const lw::Vec3& vec);
    // Combine two quaternions, if they are rotations then first rotate by q2 and then by q1
    void quatMult(lw::Quat& res, const lw::Quat& q1, const lw::Quat& q2);
    void quatNormalize(lw::Quat& res);
}
}
