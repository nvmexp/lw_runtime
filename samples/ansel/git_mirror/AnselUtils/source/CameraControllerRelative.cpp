#include <anselutils/CameraControllerRelative.h>
#include <anselutils/internal/Math.h>
#include <lw/Vec3.inl>

namespace anselutils
{
    using namespace lw;
    using namespace internal;

    CameraControllerRelative::CameraControllerRelative(const lw::Vec3& right, const lw::Vec3& up,
        const lw::Vec3& forward) :
        m_rightAxis(right),
        m_upAxis(up),
        m_forwardAxis(forward)
    {
        m_sox = 0.0f; m_soy = 0.0f;
        m_basePosition = { 0 };
        m_baseRotation = { 0.0f, 0.0f, 0.0f, 1.0f };
        m_relativeRoll = m_relativePitch = m_relativeYaw = 0.0f;
        m_relativeForward = m_relativeRight = m_relativeUp = 0.0f;
    }

    void CameraControllerRelative::setProjection(float sox, float soy)
    {
        m_sox = sox;
        m_soy = soy;
    }

    void CameraControllerRelative::update(ansel::Camera& camera)
    {
        camera.fov = m_hfov;

        // Update the rotation by applying the relative rotation in the
        // reference frame defined by m_baseRotation:
        float pitchInRadians = m_relativePitch;
        float yawInRadians = m_relativeYaw;
        float rollInRadians = m_relativeRoll;
        float rot[3][3];
        makeRotationMatrixFromEulerAngles(rot, pitchInRadians, yawInRadians, rollInRadians);

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

        float baseilw[3][3];
        matSetAndTranspose(baseilw, base);

        float base_rot_baseilw[3][3];
        matMult(base_rot_baseilw, base_rot, baseilw);

        float cameraBase[3][3];
        makeRotationMatrixFromQuaternion(cameraBase, m_baseRotation);

        float combined[3][3];
        matMult(combined, cameraBase, base_rot_baseilw);
        makeQuaternionFromRotationMatrix(camera.rotation, combined);
        camera.projectionOffsetX = m_sox;
        camera.projectionOffsetY = m_soy;

        float cameraBaseInOurSpace[3][3];
        matMult(cameraBaseInOurSpace, combined, base);

        const lw::Vec3 baseRight = { cameraBaseInOurSpace[0][0], cameraBaseInOurSpace[1][0], cameraBaseInOurSpace[2][0] };
        const lw::Vec3 baseUp = { cameraBaseInOurSpace[0][1], cameraBaseInOurSpace[1][1], cameraBaseInOurSpace[2][1] };
        const lw::Vec3 baseForward = { cameraBaseInOurSpace[0][2], cameraBaseInOurSpace[1][2], cameraBaseInOurSpace[2][2] };
        const lw::Vec3 relativeTranslate = m_relativeRight * baseRight + m_relativeUp * baseUp + m_relativeForward * baseForward;

        camera.position = m_basePosition + relativeTranslate;
    }

    void CameraControllerRelative::setCameraBase(const ansel::Camera& camera)
    {
        m_basePosition = camera.position;
        m_baseRotation = camera.rotation;
    }

    void CameraControllerRelative::setCameraBaseAndLevelWithHorizon(const ansel::Camera& camera)
    {
        m_basePosition = camera.position;

        // All our rotations are performed in a right-handed coordinate system with Y-axis up.
        // We need to colwert these into the basis that the camera is using.  If we let B be
        // the matrix formed by the basis vectors then this transformation is simply:
        //
        // T = B*R*B^-1
        //
        // Where R is the rotation matrix in our standard space and ^-1 denotes the ilwerse.
        // In order to get to R we transform this to:
        // 
        // R = B^-1*T*B
        // 
        float t[3][3];
        makeRotationMatrixFromQuaternion(t, camera.rotation);

        // multiply with the base to extract the rotation
        const lw::Vec3& axisForward = m_forwardAxis;
        const lw::Vec3& axisUp = m_upAxis;
        const lw::Vec3& axisRight = m_rightAxis;
        float base[3][3] =
        {
            { axisRight.x, axisUp.x, axisForward.x },
            { axisRight.y, axisUp.y, axisForward.y },
            { axisRight.z, axisUp.z, axisForward.z }
        };

        float base_ilw[3][3];
        matSetAndTranspose(base_ilw, base);

        float t_base[3][3];
        matMult(t_base, t, base);

        float base_ilw_t_base[3][3];
        matMult(base_ilw_t_base, base_ilw, t_base);

        float roll, pitch, yaw;
        calcEulerAnglesFromRotationMatrix(roll, pitch, yaw, base_ilw_t_base);
        
        makeRotationMatrixFromEulerAngles(base_ilw_t_base, 0.0f, yaw, 0.0f);

        matMult(t_base, base, base_ilw_t_base);
        matMult(t, t_base, base_ilw);

        makeQuaternionFromRotationMatrix(m_baseRotation, t);
    }

    void CameraControllerRelative::setCameraRotationRelativeToBase(float rollInRadians, float pitchInRadians, float yawInRadians)
    {
        m_relativeRoll = rollInRadians;
        m_relativePitch = pitchInRadians;
        m_relativeYaw = yawInRadians;
    };

    void CameraControllerRelative::setCameraPositionRelativeToBase(float forward, float right, float up)
    {
        m_relativeForward = forward;
        m_relativeRight = right;
        m_relativeUp = up;
    }
}
