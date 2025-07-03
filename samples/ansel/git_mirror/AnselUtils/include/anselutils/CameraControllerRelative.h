#pragma once
#include <anselutils/CameraController.h>

namespace anselutils
{
    class CameraControllerRelative : public CameraController
    {
    public:
        CameraControllerRelative(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward);
#if _MSC_VER >= 1800
        virtual ~CameraControllerRelative() = default;
#endif

        // Set the base, i.e. reference frame for position and orientation that this controller
        // will use.
        void setCameraBase(const ansel::Camera& camera);
        void setCameraBaseAndLevelWithHorizon(const ansel::Camera& camera);

        // This method orients the camera by firs rolling it 'rollInRadians' around the forward vector 
        // (positive to the right, negative to the left), then pitching it by rotating the camera 
        // 'pitchInRadians' around the right vector (positive up, negative down), and finally yawing 
        // it by rotating the camera 'yawRadians' around the up vector (positive to the right, 
        // negative to the left).  NOTE: Rotations are performed in an the relative 
        // coordinate frame of the camera base. The forward, up, and right are axes defined by camera
        // specified in the setCameraBase method. The notion of up/down and left/right used in this 
        // documentation is therefore also relative to these fixed axes. Turning right means turning 
        // towards the right axes (turning left is the opposite).
        void setCameraRotationRelativeToBase(float rollInRadians, float pitchInRadians, float yawInRadians);

        // This function moves the camera relative to its base frame (set using setCameraBase).
        // This happens *after* orienting the camera according to the values set using setCameraRotationRelativeToBase.
        void setCameraPositionRelativeToBase(float forward, float right, float up);

        void setProjection(float sox, float soy);

        virtual void update(ansel::Camera& camera) override;
    private:
        float m_sox, m_soy;
        lw::Vec3 m_basePosition;
        lw::Quat m_baseRotation;
        float m_relativeRoll;
        float m_relativePitch;
        float m_relativeYaw;
        float m_relativeForward;
        float m_relativeRight;
        float m_relativeUp;
        lw::Vec3 m_rightAxis, m_upAxis, m_forwardAxis;
    };
}
