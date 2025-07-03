#include "anselutils/CameraControllerFreeGlobalRotation.h"
#include "anselutils/internal/Math.h"
#include "anselutils/Utils.h"

#include <lw/Vec3.inl>

#include <limits>

namespace anselutils
{
    using namespace lw;
    using namespace internal;

    namespace
    {
        const float kPi = 3.14159265358979323846f;
    }

    CameraControllerFreeGlobalRotation::CameraControllerFreeGlobalRotation(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward) :
        CameraControllerFree(right, up, forward)
    {
    }

    void CameraControllerFreeGlobalRotation::update(ansel::Camera& camera)
    {
        if (!m_isOriginalSet)
        {
            initializeFromOriginal(camera);
            m_isOriginalSet = true;
        }
        else
        {
            float dummyRoll;
            getYawPitchAndRoll(camera, m_yaw, m_pitch, dummyRoll);
        }

        // use only fixed time step if it's non-negative - otherwise use wall-clock time
        const float secondsSinceLastUpdate = m_fixedTimeStepInSeconds < 0.0f ? m_time.elapsed() : m_fixedTimeStepInSeconds;
        const float angularDistance = m_radiansPerSecond * secondsSinceLastUpdate * m_rotationalSpeedModifier;

        // Current rotation matrix based on previous p/y/r values
        float rot[3][3] = { 0 };
        makeRotationMatrixFromEulerAngles(rot, m_pitch, m_yaw, m_roll);

        if (!areAlmostEqual(m_pitchAdjustment, 0.0f))
        {
            m_pitchAdjustment *= angularDistance;
            m_pitch += m_pitchAdjustment;
            if (m_pitch > kPi / 2)
            {
                m_pitch = kPi / 2;
            }
            else if (m_pitch < -kPi / 2)
            {
                m_pitch = -kPi / 2;
            }
        }
        else
        {
            m_pitchAdjustment = 0.0f;
        }

        if (!areAlmostEqual(m_yawAdjustment, 0.0f))
        {
            m_yawAdjustment *= angularDistance;
            m_yaw += m_yawAdjustment;
            if (m_yaw > 2 * kPi)
            {
                m_yaw -= 2 * kPi;
            }
            else if (m_yaw < -2 * kPi)
            {
                m_yaw += 2 * kPi;
            }
        }
        else 
        {
            m_yawAdjustment = 0.0f;
        }

        if (!areAlmostEqual(m_rollAdjustment, 0.0f))
        {
            m_rollAdjustment *= angularDistance;
            m_roll += m_rollAdjustment;
            clampCameraRollRadians();
        }
        else
        {
            m_rollAdjustment = 0.0f;
        }

        // Matrix used to adjust the old rotation with the new changes in the p/y/r values
        float adjRot[3][3] = { 0 };
        makeRotationMatrixFromEulerAngles(adjRot, m_pitchAdjustment, m_yawAdjustment, m_rollAdjustment);

        // All our rotations are performed in a right-handed coordinate system with Y-axis up.
        // We need to colwert these into the basis that the camera is using.  If we let B be
        // the matrix formed by the basis vectors then this transformation is simply:
        //
        // T = B*R*B^-1
        //
        // Where R is the rotation matrix in our standard space and ^-1 denotes the ilwerse.
        // Now, to callwlate the final orientation of the camera in the space that the game
        // is using we apply T to the default non-oriented camera, which is essentially the
        // basis vectors of the camera, hence B. So we get:
        //
        // T*B = B*R*B^-1*B = B*R
        //
        // So we only need to apply the basis matrix once to our rotation matrix and then
        // the vectors of the resulting matrix will be the new world space vectors of the camera:

        const lw::Vec3& axisForward = m_forwardAxis;
        const lw::Vec3& axisUp = m_upAxis;
        const lw::Vec3& axisRight = m_rightAxis;

        float base[3][3] =
        {
            { axisRight.x, axisUp.x, axisForward.x },
            { axisRight.y, axisUp.y, axisForward.y },
            { axisRight.z, axisUp.z, axisForward.z }
        }; 

        float base_rot[3][3] = { 0 };
        float rot_adjRot[3][3] = { 0 };
        matMult(rot_adjRot, rot, adjRot);
        matMult(base_rot, base, rot_adjRot);

        // reset camera acceleration if camera doesn't move
        if (m_accelerationEnabled)
        {
            // Handle camera acceleration
            if (m_translationalSpeedModifier > 1.0f) {
                m_accelerationFactor *= 1.001f;
            }
            else
            {
                m_accelerationFactor = 1.0f;
            }
                
            if (areAlmostEqual(m_rightTranslation, 0.0f) &&
                areAlmostEqual(m_upTranslation, 0.0f) &&
                areAlmostEqual(m_forwardTranslation, 0.0f))
            {
                m_accelerationFactor = 1.0f;
            }
                
            // reset camera acceleration if camera suddenly changes direction
            if ((m_prevRightTranslation * m_rightTranslation < 0.0f) ||
                (m_prevUpTranslation * m_upTranslation < 0.0f) ||
                (m_prevForwardTranslation * m_forwardTranslation < 0.0f))
            {
                m_accelerationFactor = 1.0f;
            }

            m_prevRightTranslation = m_rightTranslation;
            m_prevUpTranslation = m_upTranslation;
            m_prevForwardTranslation = m_forwardTranslation;
        }

        // Callwlate new position for camera
        const float distance = m_worldunitsPerSecond * secondsSinceLastUpdate * m_translationalSpeedModifier * m_accelerationFactor;
        const Vec3 worldRight = { base_rot[0][0], base_rot[1][0], base_rot[2][0] };
        Vec3 translation = (m_rightTranslation * distance) * worldRight;

        const Vec3 worldUp = { base_rot[0][1], base_rot[1][1], base_rot[2][1] };
        translation = translation + (m_upTranslation * distance) * worldUp;

        const Vec3 worldForward = { base_rot[0][2], base_rot[1][2], base_rot[2][2] };
        translation = translation + (m_forwardTranslation * distance) * worldForward;

        camera.position = camera.position + translation;

        camera.fov = m_hfov;

        camera.projectionOffsetX = 0.0f;
        camera.projectionOffsetY = 0.0f;

        // Copy over the rotation part:
        float baseilw[3][3] = { 0 };
        matSetAndTranspose(baseilw, base);

        float base_rot_baseilw[3][3] = { 0 };
        matMult(base_rot_baseilw, base_rot, baseilw);

        makeQuaternionFromRotationMatrix(camera.rotation, base_rot_baseilw);

        // clear out aclwmulators:
        clearAclwmulators();
    }

    void CameraControllerFreeGlobalRotation::getYawPitchAndRoll(const ansel::Camera& cam, float& yaw_out, float& pitch_out, float& roll_out)
    {
        // Determine pitch, yaw and roll based on the original camera. 
        // The camera rotation quaternion contains the basis, i.e. it is of form
        //
        // R' = B*R*B^-1
        //
        // And we need the R matrix to extract Euler angles that make sense for us.
        // We therefore callwlate:
        //
        // R = B^-1*R'*B
        //
        float base_rot_baseilw[3][3];
        makeRotationMatrixFromQuaternion(base_rot_baseilw, cam.rotation);

        const lw::Vec3& axisForward = m_forwardAxis;
        const lw::Vec3& axisUp = m_upAxis;
        const lw::Vec3& axisRight = m_rightAxis;

        float baseilw[3][3] =
        {
            { axisRight.x, axisRight.y, axisRight.z },
            { axisUp.x, axisUp.y, axisUp.z },
            { axisForward.x, axisForward.y, axisForward.z }
        };

        float base[3][3];
        matSetAndTranspose(base, baseilw);

        float rot_baseilw[3][3];
        matMult(rot_baseilw, baseilw, base_rot_baseilw);

        float rot[3][3];
        matMult(rot, rot_baseilw, base);

        calcEulerAnglesFromRotationMatrix(roll_out, pitch_out, yaw_out, rot);
    }

    void CameraControllerFreeGlobalRotation::initializeFromOriginal(const ansel::Camera& cam)
    {
        getYawPitchAndRoll(cam, m_yaw, m_pitch, m_roll);
        m_hfov = cam.fov;
        m_isRollBeingRemoved = false;
        m_rollSpeed = 0.0f;
    }

    void CameraControllerFreeGlobalRotation::setCameraRoll(float degrees)
    {
        if (!m_isRollBeingRemoved)
            CameraControllerFree::setCameraRoll(degrees);
    }
} // end of lw namespace
