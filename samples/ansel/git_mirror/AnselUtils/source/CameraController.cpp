#define ANSEL_SDK_EXPORTS
#include <anselutils/CameraController.h>

namespace anselutils
{
    using namespace lw;
    using ansel::Camera;

    namespace
    {
        const float kPi = 3.14159265358979323846f;
    }

    CameraController::CameraController() : m_hfov(90.0f)
    {
    }

    void CameraController::reset()
    {
    }

    void CameraController::update(Camera& camera)
    {
    }

    float limitFov(float fov)
    {
        // Limit the horizontal FOV to sensible values - another option would be
        // to be able to distinguish between whether this fov came from adjust or set call
        // and apply them differently (allowing the application to limit FOV to what makes sense)
        // However, this approach makes a lot of sense since horizontal FOV is what we really need 
        // to limit and some games use vertical FOV so they would have to use different values.
        if (fov > 179.0f)
            fov = 179.0f;
        if (fov < 1.0f)
            fov = 1.0f;

        return fov;
    }

    void CameraController::adjustCameraFOV(float value) 
    { 
        m_hfov = limitFov(m_hfov + value);
    }
    
    void CameraController::setCameraFOV(float value) 
    { 
        m_hfov = limitFov(value); 
    }

    float CameraController::getCameraFOV() const 
    {
        return m_hfov;
    } 

} // end of lw namespace
