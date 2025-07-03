#pragma once
#include <anselutils/CameraControllerFree.h>
#include <anselutils/ElapsedTime.h>

namespace anselutils
{
    // This camera controller works in the following way:
    // 1. 'moveCamera*' commands are exelwted in the local frame of the camera supplied to 'update'
    //    method.  That is, moveCameraUp will move the camera along the local up vector of the 
    //    supplied camera.
    // 2. 'adjustCamera*' commands are exelwted in local frame. That is, adjustCameraPitch will
    //    pitch the camera around the local right vector of the supplied camera.
    class CameraControllerFreeLocalFrame : public CameraControllerFree
    {
    public:
        CameraControllerFreeLocalFrame(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward);
        virtual ~CameraControllerFreeLocalFrame() = default;

        virtual void update(ansel::Camera& camera) override;
    private:
        virtual void getYawPitchAndRoll(const ansel::Camera& cam, float& yaw_out, float& pitch_out, float& roll_out);
        virtual void initializeFromOriginal(const ansel::Camera& cam);
        float m_chirality;
        float m_rollFromLastUpdate;
    };
}
