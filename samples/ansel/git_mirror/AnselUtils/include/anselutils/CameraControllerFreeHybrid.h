#pragma once
#include <anselutils/CameraControllerFree.h>
#include <anselutils/ElapsedTime.h>

namespace anselutils
{
    // This camera controller works in the following way:
    // 1. 'moveCamera*' commands are exelwted in the local frame of the camera supplied to 'update'
    //    method.  That is, moveCameraUp will move the camera along the local up vector of the 
    //    supplied camera.
    // 2. 'adjustCamera*' commands are exelwted in 
    //     a) global frame, i.e. world coordinates, if no roll has been applied to the camera
    //     b) local frame, i.e. camera coordinates, if a non-zero roll has been applied
    class CameraControllerFreeHybrid : public CameraControllerFree
    {
    public:
        CameraControllerFreeHybrid(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward);
        virtual ~CameraControllerFreeHybrid() = default;

        virtual void update(ansel::Camera& camera) override;
        virtual void setCameraRoll(float degrees) override;

        virtual void restoreRoll() { m_isRollBeingRemoved = true; }

        virtual bool isControllerGlobal() { return m_isRotationGlobal; }
        virtual bool isRollBeingRemoved() { return m_isRollBeingRemoved; }

        virtual void resetRollOnceOnLookaround(bool shouldReset) { m_resetRollOnLookaround = shouldReset; }

    private:
        void projectCameraRollRadians();
        virtual void getYawPitchAndRoll(const ansel::Camera& cam, float& yaw_out, float& pitch_out, float& roll_out);
        virtual void initializeFromOriginal(const ansel::Camera& cam);
        float m_rollSpeed = 0.0f;
        bool m_isRollBeingRemoved = false;
        bool m_removeErrorRoll = false;
        bool m_resetRollOnLookaround = false;

        bool m_isRotationGlobal = true;
        bool m_targetRotationGlobal = true;
        float m_chirality;
        void updateCameraWithLocalFrameRotation(float distance, float angularDistance, ansel::Camera& camera, float secondsSinceLastUpdate);
        void updateCameraWithGlobalFrameRotation(float distance, float angularDistance, ansel::Camera& camera, float secondsSinceLastUpdate);
    };
}
