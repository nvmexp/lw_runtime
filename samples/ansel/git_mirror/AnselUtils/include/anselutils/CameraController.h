#pragma once

#include <lw/Vec3.h>
#include <ansel/Camera.h>

namespace anselutils
{
    class CameraController
    {
    public:
        CameraController();

#if _MSC_VER >= 1800
        virtual ~CameraController() = default;
#endif
        virtual void update(ansel::Camera& camera) = 0;

        virtual void reset();

        void adjustCameraFOV(float value);
        void setCameraFOV(float value);
        float getCameraFOV() const;
    protected:
        float m_hfov;
    };
}
