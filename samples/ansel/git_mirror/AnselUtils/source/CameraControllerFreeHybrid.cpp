#include "anselutils/CameraControllerFreeHybrid.h"
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

    CameraControllerFreeHybrid::CameraControllerFreeHybrid(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward) :
        CameraControllerFree(right, up, forward)
    {
        m_chirality = determineChirality(right, up, forward);
    }

    void CameraControllerFreeHybrid::updateCameraWithLocalFrameRotation(float distance, 
        float angularDistance, ansel::Camera& camera, float secondsSinceLastUpdate) 
    {
        Quat localRot = { 0.0f, 0.0f, 0.0f, 1.0f };
        
        bool removeErrorRoll = false;
        // If rollDiff is non-zero, then roll was externally modified
        float rollDiff = m_roll - m_prevRoll;
        if (!areAlmostEqual(m_rollAdjustment, 0.0f) || !areAlmostEqual(rollDiff, 0.0f))
        {
            //if (!areAlmostEqual(rollDiff, 0.0f) && areAlmostEqual(m_roll, 0.0f))
            //{
            //  removeErrorRoll = true;
            //}

            float roll = m_chirality*(m_rollAdjustment*angularDistance + rollDiff);
            m_roll += m_rollAdjustment*angularDistance;
            projectCameraRollRadians();
            makeQuaternionFromRotationAxisAngle(localRot, m_forwardAxis, roll);
        }

        if (!areAlmostEqual(m_pitchAdjustment, 0.0f))
        {
            // Pitch movement detected - zero out display roll
            m_roll = 0.0f;

            // If Ansel was initialized with roll - restore it the first time user looks around
            if (m_resetRollOnLookaround)
            {
                m_isRollBeingRemoved = true;
                m_resetRollOnLookaround = false;
            }

            float pitch = m_chirality*m_pitchAdjustment*angularDistance;
            Quat pitchQuat;
            makeQuaternionFromRotationAxisAngle(pitchQuat, m_rightAxis, pitch);
            quatMult(localRot, pitchQuat, localRot);
        }

        if (!areAlmostEqual(m_yawAdjustment, 0.0f))
        {
            // Yaw movement detected - zero out display roll
            m_roll = 0.0f;

            // If Ansel was initialized with roll - restore it the first time user looks around
            if (m_resetRollOnLookaround)
            {
                m_isRollBeingRemoved = true;
                m_resetRollOnLookaround = false;
            }

            float yaw = -m_chirality*m_yawAdjustment*angularDistance;
            Quat yawQuat;
            makeQuaternionFromRotationAxisAngle(yawQuat, m_upAxis, yaw);
            quatMult(localRot, yawQuat, localRot);
        }

        quatMult(camera.rotation, camera.rotation, localRot);
        quatNormalize(camera.rotation);

        float rot[3][3];
        makeRotationMatrixFromQuaternion(rot, camera.rotation);

        const lw::Vec3& axisForward = m_forwardAxis;
        const lw::Vec3& axisUp = m_upAxis;
        const lw::Vec3& axisRight = m_rightAxis;

        float base[3][3] =
        {
            { axisRight.x, axisUp.x, axisForward.x },
            { axisRight.y, axisUp.y, axisForward.y },
            { axisRight.z, axisUp.z, axisForward.z }
        };


        float rot_base[3][3];
        matMult(rot_base, rot, base);

        auto dot = [](const lw::Vec3 & vec1, const lw::Vec3 & vec2) -> float
        {
            return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
        };
        auto sign = [](float val) -> int
        {
            return val < 0.0 ? -1 : 1;
        };

        if (m_isRollBeingRemoved || m_removeErrorRoll)
        {
            m_roll = 0.0f;

            Vec3 worldRight = { rot_base[0][0], rot_base[1][0], rot_base[2][0] };
            Vec3 worldForward = { rot_base[0][2], rot_base[1][2], rot_base[2][2] };
            Vec3 worldUp = { rot_base[0][1], rot_base[1][1], rot_base[2][1] };

            float compensate = dot(worldRight, axisUp);

            if (m_removeErrorRoll && std::abs(compensate) < 1e-5f)
            {
                m_isRollBeingRemoved = true;
            }

            if (m_isRollBeingRemoved)
            {
                if (dot(axisUp, worldUp) < -0.9f && std::abs(compensate) < 1e-5f)
                {
                    // 180 deg, need to pick arbitrary side to start restoring
                    compensate = 1.0f;
                }

                if (std::abs(compensate) < 1e-5f)
                {
                    m_isRollBeingRemoved = false;
                    m_targetRotationGlobal = true;
                }
                
                //else
                {
                    const float maxRotationSpeed = 1.0f;
                    if (compensate > maxRotationSpeed * secondsSinceLastUpdate)
                        compensate = maxRotationSpeed * secondsSinceLastUpdate;
                    if (compensate < -maxRotationSpeed * secondsSinceLastUpdate)
                        compensate = -maxRotationSpeed * secondsSinceLastUpdate;

                    Quat compensationRot = { 0.0f, 0.0f, 0.0f, 1.0f };

                    float roll = m_chirality*compensate;
                    makeQuaternionFromRotationAxisAngle(compensationRot, axisForward, roll);

                    quatMult(camera.rotation, camera.rotation, compensationRot);

                    makeRotationMatrixFromQuaternion(rot, camera.rotation);
                    matMult(rot_base, rot, base);
                }
            }

            m_removeErrorRoll = false;
        }
        // We need one frame delay for the current roll difference to be applied
        //  otherwise, compensate is too big for the threshold-leveling code to be exelwted
        m_removeErrorRoll = removeErrorRoll;

        m_prevRoll = m_roll;

        // Callwlate new position for camera
        Vec3 worldRight = { rot_base[0][0], rot_base[1][0], rot_base[2][0] };
        Vec3 translation = (m_rightTranslation*distance)*worldRight;

        Vec3 worldUp = { rot_base[0][1], rot_base[1][1], rot_base[2][1] };
        translation = translation + (m_upTranslation*distance)*worldUp;

        Vec3 worldForward = { rot_base[0][2], rot_base[1][2], rot_base[2][2] };
        translation = translation + (m_forwardTranslation*distance)*worldForward;

        camera.position = camera.position + translation;
    }

    void CameraControllerFreeHybrid::updateCameraWithGlobalFrameRotation(float distance, 
        float angularDistance, ansel::Camera& camera, float secondsSinceLastUpdate)
    {
        if (!areAlmostEqual(m_pitchAdjustment, 0.0f))
        {
            m_pitch += m_pitchAdjustment*angularDistance;
            if (m_pitch > kPi / 2)
            {
                m_pitch = kPi / 2;
            }
            else if (m_pitch < -kPi / 2)
            {
                m_pitch = -kPi / 2;
            }
        }

        if (!areAlmostEqual(m_yawAdjustment, 0.0f))
        {
            m_yaw += m_yawAdjustment*angularDistance;
            if (m_yaw > 2 * kPi)
            {
                m_yaw -= 2 * kPi;
            }
            else if (m_yaw < -2 * kPi)
            {
                m_yaw += 2 * kPi;
            }
        }

        // Direct roll control is allowed
        if (!areAlmostEqual(m_rollAdjustment, 0.0f))
        {
            m_roll += m_rollAdjustment*angularDistance;
            clampCameraRollRadians();
            m_targetRotationGlobal = false;
        }
        else if (!areAlmostEqual(m_roll, 0.0f))
        {
            m_targetRotationGlobal = false;
        }
        else
        {
            // The controller stays global frame
            if (m_resetRollOnLookaround)
            {
                // This means Ansel was initialized without roll - there is no need to reset in this case
                //  the logic should only be exelwted here and not in the case if target would be the local frame
                //  since first hybrid controller takes global frame path and only if it detects roll - switches to local frame
                m_resetRollOnLookaround = false;
            }
        }

        float rot[3][3];
        makeRotationMatrixFromEulerAngles(rot, m_pitch, m_yaw, m_roll);

        m_prevRoll = m_roll;

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


        float base_rot[3][3];
        matMult(base_rot, base, rot);

        // Callwlate new position for camera
        Vec3 worldRight = { base_rot[0][0], base_rot[1][0], base_rot[2][0] };
        Vec3 translation = (m_rightTranslation*distance)*worldRight;

        Vec3 worldUp = { base_rot[0][1], base_rot[1][1], base_rot[2][1] };
        translation = translation + (m_upTranslation*distance)*worldUp;

        Vec3 worldForward = { base_rot[0][2], base_rot[1][2], base_rot[2][2] };
        translation = translation + (m_forwardTranslation*distance)*worldForward;

        camera.position = camera.position + translation;

        // Copy over the rotation part:
        float baseilw[3][3];
        matSetAndTranspose(baseilw, base);

        float base_rot_baseilw[3][3];
        matMult(base_rot_baseilw, base_rot, baseilw);
        makeQuaternionFromRotationMatrix(camera.rotation, base_rot_baseilw);

        quatNormalize(camera.rotation);
    }

    void CameraControllerFreeHybrid::update(ansel::Camera& camera)
    {
        if (!m_isOriginalSet)
        {
            initializeFromOriginal(camera);
            if (m_targetRotationGlobal)
                m_isRotationGlobal = true;
            else
                m_isRotationGlobal = false;

            m_isOriginalSet = true;
        }
        else
        {
            float dummyRoll;
            getYawPitchAndRoll(camera, m_yaw, m_pitch, dummyRoll);
        }

        float secondsSinceLastUpdate;
        // use only fixed time step if it's non-negative - otherwise use wall-clock time
        if (m_fixedTimeStepInSeconds < 0.0f)
            secondsSinceLastUpdate = m_time.elapsed();
        else
            secondsSinceLastUpdate = m_fixedTimeStepInSeconds;

        float angularDistance = m_radiansPerSecond * secondsSinceLastUpdate * m_rotationalSpeedModifier;
        // reset camera acceleration if camera doesn't move
        if (m_accelerationEnabled)
        {
            // Handle camera acceleration
            if (m_translationalSpeedModifier > 1.0f)
                m_accelerationFactor *= 1.001f;
            else
                m_accelerationFactor = 1.0f;

            if (areAlmostEqual(m_rightTranslation, 0.0f) &&
                areAlmostEqual(m_upTranslation, 0.0f) &&
                areAlmostEqual(m_forwardTranslation, 0.0f))
                m_accelerationFactor = 1.0f;

            // reset camera acceleration if camera suddenly changes direction
            if ((m_prevRightTranslation * m_rightTranslation < 0.0f) ||
                (m_prevUpTranslation * m_upTranslation < 0.0f) ||
                (m_prevForwardTranslation * m_forwardTranslation < 0.0f))
                m_accelerationFactor = 1.0f;

            m_prevRightTranslation = m_rightTranslation;
            m_prevUpTranslation = m_upTranslation;
            m_prevForwardTranslation = m_forwardTranslation;
        }
        float distance = m_worldunitsPerSecond * secondsSinceLastUpdate * 
            m_translationalSpeedModifier * m_accelerationFactor;

        if (m_isRotationGlobal)
            updateCameraWithGlobalFrameRotation(distance, angularDistance, camera, secondsSinceLastUpdate);
        else
            updateCameraWithLocalFrameRotation(distance, angularDistance, camera, secondsSinceLastUpdate);

        camera.fov = m_hfov;

        camera.projectionOffsetX = 0.0f;
        camera.projectionOffsetY = 0.0f;

        // clear out aclwmulators:
        clearAclwmulators();

        bool wasRotationGlobal = m_isRotationGlobal;
        if (m_targetRotationGlobal)
            m_isRotationGlobal = true;
        else
            m_isRotationGlobal = false;

        if (!wasRotationGlobal && m_isRotationGlobal)
        {
            initializeFromOriginal(camera);
            m_roll = 0.0f;
            m_prevRoll = m_roll;
        }
    }

    void CameraControllerFreeHybrid::projectCameraRollRadians()
    {
        while (m_roll > kPi)
        {
            m_roll -= 2.f * kPi;
        }

        while (m_roll < -kPi)
        {
            m_roll += 2.f * kPi;
        }
    }

    void CameraControllerFreeHybrid::getYawPitchAndRoll(const ansel::Camera& cam, float& yaw_out, float& pitch_out, float& roll_out)
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

    void CameraControllerFreeHybrid::initializeFromOriginal(const ansel::Camera& cam)
    {
        getYawPitchAndRoll(cam, m_yaw, m_pitch, m_roll);
        m_prevRoll = m_roll;

        m_hfov = cam.fov;
        m_isRollBeingRemoved = false;
        m_rollSpeed = 0.0f;
    }

    void CameraControllerFreeHybrid::setCameraRoll(float degrees)
    {
        if (!m_isRollBeingRemoved)
            CameraControllerFree::setCameraRoll(degrees);
    }
} // end of lw namespace
