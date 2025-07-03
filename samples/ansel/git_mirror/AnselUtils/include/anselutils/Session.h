#pragma once
#include <ansel/Defines.h>
#include <ansel/Configuration.h>
#include <ansel/Camera.h>

namespace anselutils
{
    class CameraController;

    class Session
    {
    public:
        Session(const ansel::Configuration& cfg, uint32_t viewportWidth, uint32_t viewportHeight);
        ~Session();

        // Set the camera controller to use
        void setCameraController(CameraController* camControl);
        // Get the camera controller lwrrently in use
        CameraController* getCameraController() const;
        // Set camera explicitly
        void setDisplayCamera(ansel::Camera& camera);
        // Get latest camera
        ansel::Camera getDisplayCamera() const;
        // Get original camera
        ansel::Camera getOriginalCamera() const;
        // Revert back to the camera settings at the start of the session
        void restoreOriginalCameraSettings();
        // Was updateCamera called at least once for this object?
        bool isDisplayCameraInitialized() const;
        // Update camera using underlying camera controller
        void updateCamera(ansel::Camera& cam);
        // Return viewport width
        uint32_t getViewportWidth() const;
        // Return viewport height
        uint32_t getViewportHeight() const;
    private:
        bool m_isAwaitingOriginalCamera = true;
        bool m_isRestoreRequested = false;
        bool m_useExternalCamera = false;
        bool m_isDisplayCameraInitialized = false;
        anselutils::CameraController* m_camController = nullptr;
        ansel::Camera m_originalCamera;
        ansel::Camera m_displayCamera;
        ansel::Configuration m_configuration;
        uint32_t m_viewportWidth = 0, m_viewportHeight = 0;
    };
}
