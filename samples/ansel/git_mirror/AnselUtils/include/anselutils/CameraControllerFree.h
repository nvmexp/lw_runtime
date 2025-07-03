#pragma once
#include <anselutils/CameraController.h>
#include <anselutils/ElapsedTime.h>

namespace anselutils
{
    class CameraControllerFree : public CameraController
    {
    public:
        CameraControllerFree(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward);
        virtual ~CameraControllerFree() = default;

        // All input values should be in the range [-1,1] (will be colwerted to actual movement by this controller based on speed settings)
        void moveCameraRight(float value) { m_rightTranslation += value; }
        void moveCameraForward(float value) { m_forwardTranslation += value; }
        void moveCameraUp(float value) { m_upTranslation += value; }

        void adjustCameraYaw(float value) { m_yawAdjustment += value; }
        void adjustCameraPitch(float value) { m_pitchAdjustment += value; }
        void adjustCameraRoll(float value) { m_rollAdjustment += value; }

        virtual void setCameraRoll(float degrees);
        float getCameraRoll() const;

        void setTranslationalSpeed(float worldUnitsPerSecond);
        void setTranslationalSpeedMultiplier(float value);

        float getTranslationalSpeed() const;
        float getTranslationalSpeedMultiplier() const;

        void setRotationalSpeed(float degreesPerSecond);
        void setRotationalSpeedMultiplier(float value);

        float getRotationalSpeed() const;
        float getRotationalSpeedMultiplier() const;

        virtual void update(ansel::Camera& camera) override = 0;
        virtual void reset() override;

        void setAccelerationMode(bool mode);
        bool getAccelerationMode() const;

        // For unit tests
        // Set this value to a non-negative value to get fixed time step behavior
        void setFixedTimeStep(float timeInSeconds);
    protected:
        virtual void getYawPitchAndRoll(const ansel::Camera& cam, float& yaw_out, float& pitch_out, float& roll_out) = 0;
        virtual void initializeFromOriginal(const ansel::Camera& cam) = 0;

        void clearAclwmulators();
        void clampCameraRollRadians();
        float m_worldunitsPerSecond;
        float m_radiansPerSecond;
        float m_translationalSpeedModifier;
        float m_rotationalSpeedModifier;
        lw::Vec3 m_rightAxis, m_upAxis, m_forwardAxis;
        // These are all relative to the current display camera
        float m_rightTranslation;
        float m_forwardTranslation;
        float m_upTranslation;
        // These are all relative to the current display camera (least confusing):
        float m_yawAdjustment;
        float m_pitchAdjustment;
        float m_rollAdjustment;
        float m_accelerationFactor = 1.0f;
        float m_prevRightTranslation = 0.0f;
        float m_prevForwardTranslation = 0.0f;
        float m_prevUpTranslation = 0.0f;
        float m_yaw, m_pitch, m_roll;
        float m_prevRoll;
        bool m_isOriginalSet;
        ElapsedTime m_time;
        float m_fixedTimeStepInSeconds;
        // Acceleration mode
        bool m_accelerationEnabled;
    };
}
