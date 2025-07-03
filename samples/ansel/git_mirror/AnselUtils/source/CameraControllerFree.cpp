#include "anselutils/CameraControllerFree.h"
#include "anselutils/internal/Math.h"
#include "anselutils/Utils.h"

#include <lw/Vec3.inl>

#include <limits>

namespace anselutils
{
    using namespace lw;

    namespace
    {
        const float kPi = 3.14159265358979323846f;
    }

    CameraControllerFree::CameraControllerFree(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward) :
        CameraController(),
        m_rightTranslation(0),
        m_forwardTranslation(0),
        m_upTranslation(0),
        m_yawAdjustment(0),
        m_pitchAdjustment(0),
        m_rollAdjustment(0),
        m_accelerationFactor(1.0f),
        m_prevRightTranslation(0.0f),
        m_prevForwardTranslation(0.0f),
        m_prevUpTranslation(0.0f),
        m_yaw(0),
        m_pitch(0),
        m_roll(0),
        m_translationalSpeedModifier(1.0f),
        m_rotationalSpeedModifier(1.0f),
        m_isOriginalSet(false),
        m_time(0.033f), // 33ms is maximum amount of time for movement callwlation
        m_worldunitsPerSecond(1.0f),
        m_radiansPerSecond(kPi/4.0f),
        m_fixedTimeStepInSeconds(-1.0f),
        m_accelerationEnabled(true)
    {
        m_rightAxis = right;
        m_upAxis = up;
        m_forwardAxis = forward;
    }

    void CameraControllerFree::setAccelerationMode(bool mode) { m_accelerationEnabled = mode; }
    bool CameraControllerFree::getAccelerationMode() const { return m_accelerationEnabled; }
    float CameraControllerFree::getTranslationalSpeed() const { return m_worldunitsPerSecond; }
    float CameraControllerFree::getTranslationalSpeedMultiplier() const { return m_translationalSpeedModifier; }
    float CameraControllerFree::getRotationalSpeed() const { return m_radiansPerSecond; }
    float CameraControllerFree::getRotationalSpeedMultiplier() const { return m_rotationalSpeedModifier; }

    void CameraControllerFree::setCameraRoll(float degrees)
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
    float CameraControllerFree::getCameraRoll() const
    {
        return m_roll * 180.0f / kPi;
    }

    void CameraControllerFree::reset()
    {
        m_isOriginalSet = false;
        clearAclwmulators();
    }

    void CameraControllerFree::update(ansel::Camera& camera)
    {
        // pure virtual call shouldn't happen
    }

    void CameraControllerFree::getYawPitchAndRoll(const ansel::Camera& cam, float& yaw_out, float& pitch_out, float& roll_out)
    {
        // pure virtual call shouldn't happen
    }

    void CameraControllerFree::initializeFromOriginal(const ansel::Camera& cam)
    {
        // pure virtual call shouldn't happen
    }

    void CameraControllerFree::clearAclwmulators()
    {
        m_rightTranslation = m_forwardTranslation = m_upTranslation = 0.0f;
        m_yawAdjustment = m_pitchAdjustment = m_rollAdjustment = 0.0f;
    }

    void CameraControllerFree::clampCameraRollRadians()
    {
        if (m_roll > kPi)
        {
            m_roll = kPi;
        }
        else if (m_roll < -kPi)
        {
            m_roll = -kPi;
        }
    }

    void CameraControllerFree::setRotationalSpeed(float degreesPerSecond)
    {
        m_radiansPerSecond = degreesPerSecond*kPi / 180.0f;
    }


    void CameraControllerFree::setRotationalSpeedMultiplier(float value)
    {
        m_rotationalSpeedModifier = value;
    }

    void CameraControllerFree::setTranslationalSpeed(float worldUnitsPerSecond)
    {
        m_worldunitsPerSecond = worldUnitsPerSecond;
    }

    void CameraControllerFree::setTranslationalSpeedMultiplier(float value)
    {
        m_translationalSpeedModifier = value;
    }

    void CameraControllerFree::setFixedTimeStep(float timeStepInSeconds)
    {
        m_fixedTimeStepInSeconds = timeStepInSeconds;
    }
} // end of lw namespace
