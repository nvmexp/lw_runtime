#pragma once
#include <anselutils/CameraControllerFree.h>
#include <anselutils/ElapsedTime.h>

namespace anselutils
{
    // This camera controller works in the following way:
    // 1. 'moveCamera*' commands are exelwted in the local frame of the camera supplied to 'update'
    //    method.  That is, moveCameraUp will move the camera along the local up vector of the 
    //    supplied camera.
    // 2. 'adjustCamera*' commands are exelwted in the global frame, i.e. world coordinates. This
    //    means that an adjustCameraYaw will always result in rotation around the world up vector.
    // 3. Finally, in order to prevent the user from becoming disoriented any roll is removed
    //    when the camera is yawed or pitched. This means that roll needs to be applied last to
    //    have an effect and it will be smoothly reset to zero when the camera receives yaw or
    //    pitch commands.
    class CameraControllerFreeGlobalRotation : public CameraControllerFree
    {
    public:
        CameraControllerFreeGlobalRotation(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward);
        virtual ~CameraControllerFreeGlobalRotation() = default;

        virtual void update(ansel::Camera& camera) override;
        virtual void setCameraRoll(float degrees) override;
    private:
        virtual void getYawPitchAndRoll(const ansel::Camera& cam, float& yaw_out, float& pitch_out, float& roll_out);
        virtual void initializeFromOriginal(const ansel::Camera& cam);
        float m_rollSpeed = 0.0f;
        bool m_isRollBeingRemoved = false;
    };
}
