#include "anselutils/CameraControllerFreeLocalFrame.h"
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

    CameraControllerFreeLocalFrame::CameraControllerFreeLocalFrame(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward) :
        CameraControllerFree(right, up, forward),
        m_rollFromLastUpdate(0.0f)
    {
        m_chirality = determineChirality(right, up, forward);
    }

    void CameraControllerFreeLocalFrame::update(ansel::Camera& camera)
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

        // Let's now perform rotations (if needed):
        float secondsSinceLastUpdate;
        // use only fixed time step if it's non-negative - otherwise use wall-clock time
        if (m_fixedTimeStepInSeconds < 0.0f)
            secondsSinceLastUpdate = m_time.elapsed();
        else
            secondsSinceLastUpdate = m_fixedTimeStepInSeconds;

        float angularDistance = m_radiansPerSecond * secondsSinceLastUpdate * m_rotationalSpeedModifier;

        Quat localRot = { 0.0f, 0.0f, 0.0f, 1.0f };
        m_rollAdjustment = m_roll - m_rollFromLastUpdate;
        if (!areAlmostEqual(m_rollAdjustment, 0.0f))
        {
            float roll = m_chirality*m_rollAdjustment;
            makeQuaternionFromRotationAxisAngle(localRot, m_forwardAxis, roll);
            m_rollFromLastUpdate = m_roll;
        }

        if (!areAlmostEqual(m_pitchAdjustment, 0.0f))
        {
            float pitch = m_chirality*m_pitchAdjustment*angularDistance;
            Quat pitchQuat;
            makeQuaternionFromRotationAxisAngle(pitchQuat, m_rightAxis, pitch);
            quatMult(localRot, pitchQuat, localRot);
        }

        if (!areAlmostEqual(m_yawAdjustment, 0.0f))
        {
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

        // Callwlate new position for camera
        float distance = m_worldunitsPerSecond * secondsSinceLastUpdate * m_translationalSpeedModifier;
        Vec3 worldRight = { rot_base[0][0], rot_base[1][0], rot_base[2][0] };
        Vec3 translation = (m_rightTranslation*distance)*worldRight;

        Vec3 worldUp = { rot_base[0][1], rot_base[1][1], rot_base[2][1] };
        translation = translation + (m_upTranslation*distance)*worldUp;

        Vec3 worldForward = { rot_base[0][2], rot_base[1][2], rot_base[2][2] };
        translation = translation + (m_forwardTranslation*distance)*worldForward;

        camera.position = camera.position + translation;

        camera.fov = m_hfov;

        camera.projectionOffsetX = 0.0f;
        camera.projectionOffsetY = 0.0f;

        // clear out aclwmulators:
        clearAclwmulators();
    }

    void CameraControllerFreeLocalFrame::getYawPitchAndRoll(const ansel::Camera& cam, float& yaw_out, float& pitch_out, float& roll_out)
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

    void CameraControllerFreeLocalFrame::initializeFromOriginal(const ansel::Camera& cam)
    {
        getYawPitchAndRoll(cam, m_yaw, m_pitch, m_roll);
        m_rollFromLastUpdate = m_roll;

        m_hfov = cam.fov;
    }
} // end of lw namespace
