#pragma once
#include <anselutils/CameraController.h>
#include <anselutils/ElapsedTime.h>

namespace anselutils
{
    using ansel::Camera;

    class CameraControllerOrbit : public CameraController
    {
    public:
        CameraControllerOrbit(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward);
#if _MSC_VER >= 1800
        virtual ~CameraControllerOrbit() = default;
#endif

        void moveOrbitCenterRight(float value) { m_centerRightAdjustment += value; }
        void moveOrbitCenterForward(float value) { m_centerForwardAdjustment += value; }
        void moveOrbitCenterUp(float value) { m_centerUpAdjustment += value; }

        void adjustOrbitYaw(float value) { m_yawAdjustment += value; }
        void adjustOrbitPitch(float value) { m_pitchAdjustment += value; }
        void adjustCameraRoll(float value) { m_rollAdjustment += value; }

        void setCameraRoll(float degrees);

        void adjustOrbitDistance(float value) { m_distanceAdjustment += value; }

        virtual void update(Camera& camera) override;

    private:
        lw::Vec3 m_center;
        // These are all relative to a fixed spherical coordinates system (least confusing):
        float m_yawAdjustment;
        float m_pitchAdjustment;
        float m_rollAdjustment;
        float m_centerForwardAdjustment;
        float m_centerRightAdjustment;
        float m_centerUpAdjustment;
        float m_distanceAdjustment;
        // absolute values
        float m_orbitDistance;
        float m_yaw;
        float m_pitch;
        float m_roll;
        float m_forward;
        float m_right;
        float m_up;
        ElapsedTime m_time;
        lw::Vec3 m_rightAxis, m_upAxis, m_forwardAxis;

        void clampCameraRollRadians();
    };

}
