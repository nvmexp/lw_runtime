#define ANSEL_SDK_EXPORTS
#include <ansel/Configuration.h>
#include <ansel/Camera.h>
#include <lw/Vec3.inl>
#include <string>
#include <windows.h> // For GetModuleHandleA

namespace ansel
{
    namespace
    {
        decltype(updateCamera)* s_updateCameraFunc = nullptr;
        bool s_anselIsAvailable = true;
        Configuration s_config;
        std::string s_titleName; // so we can manage the lifetime ourselves

        bool isNullVector(const lw::Vec3& v)
        {
            const float nullThreshold = 0.000001f;
            if (fabs(v.x) < nullThreshold && fabs(v.y) < nullThreshold && fabs(v.z) < nullThreshold)
                return true;

            return false;
        }

        bool isOrthogonal(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward)
        {
            const float orthogonalityThreshold = 0.000001f;
            if (fabs(lw::vecDot(right, up)) > orthogonalityThreshold ||
                fabs(lw::vecDot(right, forward)) > orthogonalityThreshold ||
                fabs(lw::vecDot(up, forward)) > orthogonalityThreshold)
                return false;

            return true;
        }

        void makeQuaternionFromRotationMatrix(lw::Quat& q, const float r[3][3])
        {
            float trace = r[0][0] + r[1][1] + r[2][2];

            if (trace > 0.0f) {
                // |w| > 1/2, may as well choose w > 1/2
                float s = sqrtf(trace + 1.0f);  // 2w

                q.w = s * 0.5f;
                s = 0.5f / s;       // 1/(4w)

                q.x = (r[2][1] - r[1][2]) * s;
                q.y = (r[0][2] - r[2][0]) * s;
                q.z = (r[1][0] - r[0][1]) * s;
            }
            else {
                // |w| <= 1/2
                int i = 0;
                if (r[1][1] > r[0][0]) i = 1;
                if (r[2][2] > r[i][i]) i = 2;

                int j = (1 << i) & 3; // i + 1 modulo 3.
                int k = (1 << j) & 3;

                float s = sqrtf(r[i][i] - r[j][j] - r[k][k] + 1.0f);

                float* component = &q.x;
                component[i] = s * 0.5f;
                s = 0.5f / s;
                component[j] = (r[i][j] + r[j][i]) * s;
                component[k] = (r[k][i] + r[i][k]) * s;
                q.w = (r[k][j] - r[j][k]) * s;
            }
        }

        void makeRotationMatrixFromQuaternion(float rot[3][3], const lw::Quat& q)
        {
            float s = 2.0f;

            float xs = q.x * s;
            float ys = q.y * s;
            float zs = q.z * s;

            float wx = q.w * xs;
            float wy = q.w * ys;
            float wz = q.w * zs;

            float xx = q.x * xs;
            float xy = q.x * ys;
            float xz = q.x * zs;

            float yy = q.y * ys;
            float yz = q.y * zs;
            float zz = q.z * zs;

            rot[0][0] = 1.0f - (yy + zz);
            rot[0][1] = xy - wz;
            rot[0][2] = xz + wy;

            rot[1][0] = xy + wz;
            rot[1][1] = 1.0f - (xx + zz);
            rot[1][2] = yz - wx;

            rot[2][0] = xz - wy;
            rot[2][1] = yz + wx;
            rot[2][2] = 1.0f - (xx + yy);
        }
    }

    SetConfigurationStatus setConfiguration(const Configuration& config)
    {
        if (config.sdkVersion != ANSEL_SDK_VERSION)
        {
            s_anselIsAvailable = false;
            return kSetConfigurationIncompatibleVersion;
        }

            // check if the basis given is orthogonal
        if (isNullVector(config.right) ||
            isNullVector(config.up) ||
            isNullVector(config.forward) ||
            !isOrthogonal(config.right, config.up, config.forward) ||
            // check translational and rotational speed multipliers
            (config.translationalSpeedInWorldUnitsPerSecond == 0.0) ||
            (config.rotationalSpeedInDegreesPerSecond == 0.0) ||
            // check if fov type is correct
            (config.fovType != kHorizontalFov && config.fovType != kVerticalFov) ||
            // check if necessary callbacks are set
            (config.startSessionCallback == nullptr || config.stopSessionCallback == nullptr) ||
            // check if the window handle is set
            (config.gameWindowHandle == nullptr))
        {
            s_anselIsAvailable = false;
            return kSetConfigurationIncorrectConfiguration;
        }

        // in case we have a correct configuration
        s_config = config;
        // we never trust the lifetime of the titleName passed in, we make a copy if needed
        if (config.titleNameUtf8)
        {
            s_titleName = config.titleNameUtf8; 
            s_config.titleNameUtf8 = s_titleName.c_str();
        }
        s_anselIsAvailable = true;
        return kSetConfigurationSuccess;
    }

    void updateCamera(ansel::Camera& cam)
    {
        if (s_updateCameraFunc)
            s_updateCameraFunc(cam);
    }

    bool isAnselAvailable()
    {
        // search for LwCamera32/64 DLL in the process
        const char* moduleName =
#if _M_AMD64
            "LwCamera64.dll";
#else
            "LwCamera32.dll";
#endif

        // in case s_anselIsAvailable was set to false (version mismatch or incorrect configuration during setConfiguration call)
        // we will return false immediately
        return s_anselIsAvailable && (GetModuleHandleA(moduleName) != nullptr);
    }

    void quaternionToRotationMatrixVectors(const lw::Quat& q, lw::Vec3& right, lw::Vec3& up, lw::Vec3& forward)
    {
        float rot[3][3];
        makeRotationMatrixFromQuaternion(rot, q);
        right = { rot[0][0], rot[1][0], rot[2][0] };
        up = { rot[0][1], rot[1][1], rot[2][1] };
        forward = { rot[0][2], rot[1][2], rot[2][2] };
    }

    void rotationMatrixVectorsToQuaternion(const lw::Vec3& right, const lw::Vec3& up, const lw::Vec3& forward, lw::Quat& q)
    {
        const float rot[3][3] = {
            { right.x, up.x, forward.x },
            { right.y, up.y, forward.y },
            { right.z, up.z, forward.z }
        };
        makeQuaternionFromRotationMatrix(q, rot);
    }

    // These functions gets called by the driver.

    // Even though cfg object might be of different version, since we can only add things to it,
    // the binary code implementing assigment here is only going to fill the neccessary part of the object.
    // Everything beyond that will be left untouched.
    ANSEL_SDK_INTERNAL_API void getConfiguration(Configuration& cfg)
    {
        cfg = s_config;
    }

    ANSEL_SDK_INTERNAL_API void setUpdateCameraFunc(decltype(updateCamera)* updateCameraFunc)
    {
        s_updateCameraFunc = updateCameraFunc;
    }

    ANSEL_SDK_INTERNAL_API uint32_t getConfigurationSize()
    {
        return sizeof(Configuration);
    }

    ANSEL_SDK_INTERNAL_API uint32_t getSessionConfigurationSize()
    {
        return sizeof(SessionConfiguration);
    }

    ANSEL_SDK_INTERNAL_API void initializeConfiguration(Configuration& cfg)
    {
        cfg = Configuration();
    }

    ANSEL_SDK_INTERNAL_API void initializeSessionConfiguration(SessionConfiguration& sessionCfg)
    {
        sessionCfg = SessionConfiguration();
    }
}
