#define ANSEL_SDK_EXPORTS
#include <ansel/Camera.h>
#include <anselutils/Utils.h>
#include <anselutils/Session.h>
#include <anselutils/CameraController.h>
#include <lw/Vec3.inl>
#include <vector>

namespace anselutils
{
    Session::Session(const ansel::Configuration& cfg, uint32_t viewportWidth, uint32_t viewportHeight)
    {
        m_configuration = cfg;
        m_viewportWidth = viewportWidth;
        m_viewportHeight = viewportHeight;
        m_isAwaitingOriginalCamera = true;
        m_isRestoreRequested = false;
        m_camController = nullptr;
    }

    Session::~Session()
    {
        m_isDisplayCameraInitialized = false; 
    }

    void Session::restoreOriginalCameraSettings()
    {
        m_isRestoreRequested = true;
    }

    void Session::setCameraController(anselutils::CameraController* camController)
    {
        m_camController = camController;
        if (camController)
            camController->reset();
    }

    CameraController* Session::getCameraController() const
    {
        return m_camController;
    }

    bool Session::isDisplayCameraInitialized() const
    {
        return m_isDisplayCameraInitialized;
    }

    void Session::updateCamera(ansel::Camera& cam)
    {
        if (m_useExternalCamera)
        {
            cam = m_displayCamera;
            m_useExternalCamera = false;
            return;
        }

        m_displayCamera = cam;

        if (!m_isDisplayCameraInitialized)
        {
            m_isDisplayCameraInitialized = true;
        }

        if (m_isRestoreRequested)
        {
            cam = m_originalCamera;
            m_isRestoreRequested = false;
            return;
        }

        CameraController* cc = m_camController;
        if (!cc)
            return; // Nothing to do if no camera controller has been activated

        if (m_isAwaitingOriginalCamera)
        {
            m_originalCamera = cam;
            cc->reset();
            m_isAwaitingOriginalCamera = false;
        }

        if (m_configuration.fovType == ansel::kVerticalFov)
            cam.fov = static_cast<float>(colwertVerticalToHorizontalFov(cam.fov, m_viewportWidth, m_viewportHeight));
        
        cc->update(cam);

        if (m_configuration.fovType == ansel::kVerticalFov)
            cam.fov = static_cast<float>(colwertHorizontalToVerticalFov(cam.fov, m_viewportWidth, m_viewportHeight));
    }

    // This method is needed for shot sequencing
    ansel::Camera Session::getDisplayCamera() const { return m_displayCamera; }

    // This method is needed for telemetry stats collection
    ansel::Camera Session::getOriginalCamera() const { return m_originalCamera; }

    void Session::setDisplayCamera(ansel::Camera& cam)
    {
        m_displayCamera = cam;
        m_useExternalCamera = true;
    }

    uint32_t Session::getViewportWidth() const { return m_viewportWidth; }
    uint32_t Session::getViewportHeight() const { return m_viewportHeight; }
}
