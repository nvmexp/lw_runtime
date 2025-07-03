#include "anselutils/CameraControllerOrbit.h"
#include "anselutils/Utils.h"

namespace anselutils
{
    namespace
    {
        const float kPi = 3.14159265358979323846f;

        lw::Vec3 operator-(const lw::Vec3& a, const lw::Vec3& b) { return lw::Vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
        lw::Vec3 operator+(const lw::Vec3& a, const lw::Vec3& b) { return lw::Vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
        float operator*(const lw::Vec3& a, const lw::Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        lw::Vec3 operator*(const lw::Vec3& a, float multiplier) { return lw::Vec3{ a.x * multiplier, a.y * multiplier, a.z * multiplier }; }
        lw::Vec3 cross(const lw::Vec3& a, const lw::Vec3& b) { return lw::Vec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
        lw::Quat operator*(const lw::Quat& a, const lw::Quat& b)
        {
            lw::Quat result;
            result.x = a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x;
            result.y = -a.x * b.z + a.y * b.w + a.z * b.x + a.w * b.y;
            result.z = a.x * b.y - a.y * b.x + a.z * b.w + a.w * b.z;
            result.w = -a.x * b.x - a.y * b.y - a.z * b.z + a.w * b.w;
            return result;
        }
        lw::Vec3 normalize(const lw::Vec3& in)
        {
            const float l = sqrtf(in.x * in.x + in.y * in.y + in.z * in.z);
            if (fabs(l) >= 0.0f)
            {
                const float recipL = 1.0f / l;
                return lw::Vec3{ in.x * recipL, in.y * recipL, in.z * recipL };
            }
            else return in;
        }

        lw::Quat constructQuat(const lw::Vec3& axis, float angle)
        {
            const float sinAngle = sinf(angle / 2.0f);
            return lw::Quat{ axis.x * sinAngle, axis.y * sinAngle, axis.z * sinAngle, cosf(angle / 2) };
        }

        lw::Quat lookAt(const lw::Vec3& sourcePoint, const lw::Vec3& destPoint, const lw::Vec3& forward, const lw::Vec3& up)
        {
            lw::Vec3 forwardVector = normalize(destPoint - sourcePoint);

            const float dot = forward * forwardVector;

            if (fabs(dot + 1.0f) < 0.000001f)
                return lw::Quat{ up.x, up.y, up.z, kPi };
            if (fabs(dot - 1.0f) < 0.000001f)
                return lw::Quat{ 0.0f, 0.0f, 0.0f, 1.0f };

            lw::Vec3 rotAxis = cross(forward, forwardVector);
            rotAxis = normalize(rotAxis);
            return constructQuat(rotAxis, cosf(dot));
        }
    }

    CameraControllerOrbit::CameraControllerOrbit(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward) :
        m_yawAdjustment(0.0f),
        m_pitchAdjustment(0.0f),
        m_rollAdjustment(0.0f),
        m_centerForwardAdjustment(0.0f),
        m_centerRightAdjustment(0.0f),
        m_centerUpAdjustment(0.0f),
        m_distanceAdjustment(0.0f),
        m_yaw(kPi / 4.0f),
        m_pitch(kPi / 4.0f),
        m_roll(0.0f),
        m_time(0.033f) // 33ms is maximum amount of time for movement callwlation
    {
        m_rightAxis = right;
        m_upAxis = up;
        m_forwardAxis = forward; 
    }

    void CameraControllerOrbit::setCameraRoll(float degrees)
    {
        if (areAlmostEqual(degrees, 0.0f))
        {
            m_roll = 0.0f;
        }
        else
        {
            m_roll = degrees * kPi / 180.0f;
        }
        clampCameraRollRadians();
    }

    void CameraControllerOrbit::update(Camera& camera)
    {
        // integrate all the values
        const float secondsSinceLastUpdate = m_time.elapsed();
        m_orbitDistance += secondsSinceLastUpdate * m_distanceAdjustment;
        m_yaw += secondsSinceLastUpdate * m_yawAdjustment;
        m_pitch += secondsSinceLastUpdate * m_pitchAdjustment;
        m_roll += secondsSinceLastUpdate * m_rollAdjustment;
        m_forward += secondsSinceLastUpdate * m_centerForwardAdjustment;
        m_right += secondsSinceLastUpdate * m_centerRightAdjustment;
        m_up += secondsSinceLastUpdate * m_centerUpAdjustment;

        const lw::Vec3& axisForward = m_forwardAxis;
        const lw::Vec3& axisUp = m_upAxis;
        const lw::Vec3& axisRight = m_rightAxis;

        // callwlate new camera
        m_center = axisForward * m_forward + axisRight * m_right + axisUp * m_up;
        const lw::Vec3 rotVector = normalize(axisForward * cosf(m_yaw) + axisRight * sinf(m_yaw) + axisUp * sinf(m_pitch));
        camera.position = m_center + rotVector * m_orbitDistance;
        camera.rotation = constructQuat(axisForward, m_roll) * lookAt(camera.position, m_center, axisForward, axisUp);
        camera.projectionOffsetX = 0.0f;
        camera.projectionOffsetY = 0.0f;
        camera.fov = m_hfov;
    }

    void CameraControllerOrbit::clampCameraRollRadians()
    {
        if (m_roll > kPi / 2)
        {
            m_roll = kPi / 2;
        }
        else if (m_roll < -kPi / 2)
        {
            m_roll = -kPi / 2;
        }
    }
}
