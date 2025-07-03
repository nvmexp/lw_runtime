#pragma once

#include <string>
#include <vector>
#include <memory>
#include "Config.h"
#include "ErrorReporting.h"
#include "InputHandler.h"
#include "CommonStructs.h"
#include "ansel/Camera.h"
#include "ansel/Configuration.h"
#include "ansel/Hints.h"
#include "ansel/UserControls.h"
#include "anselutils/Session.h"
#include "anselutils/CameraController.h"
#include "anselutils/CameraControllerOrbit.h"
#include "anselutils/CameraControllerFree.h"
#include "anselutils/CameraControllerRelative.h"
#include "UIBase.h"

// Darkroom includes
#include "darkroom/Director.h"
#include "darkroom/PixelFormat.h"
#include "darkroom/JobProcessing.h"

struct DirectorProcJob
{
    HANDLE hndl;
    std::wstring pathFilename;
};

class AnselUI;

// TODO: introduce namespace?
struct AnselSDKCaptureParameters
{
    std::wstring intermediateFolderPath;
    std::wstring snapshotPath;
    std::wstring tagModel;
    std::wstring tagSoftware;
    std::wstring tagDrsName;
    std::wstring tagDrsProfileName;
    std::wstring tagAppTitleName;
    std::wstring tagAppCMSID;
    std::wstring tagAppShortName;
    std::wstring tagDescription;
    std::wstring tagActiveFilters;
    const wchar_t* appName;
    ShotType shotToTake;
    bool restoreRoll;
    bool makeScreenshot;
    bool keepIntermediateShots;
    bool forceLosslessSuperRes;
    bool forceLossless360;
    bool isShotHDR;                                             // This flag should only be set if user specifically selected "Save as HDR"
    bool isShotHDRJXR;                                          // This flag is set when we want to export the HDR capture with JXR file format, it should never be set when isShotHDR is set (which ensures EXR export)
    bool isShotRawHDR;                                          // This flag is only set when the shot is supposed to capture the raw HDR buffer. This is decided internally, and not by the UI.
    bool isShotDisplayHDR;                                      // This flag is only set when the shot is supposed to capture the presentable HDR buffer. This is decided internally, and not by the UI.
    bool isShotPreviewRequired;                                 // TODO avoroshilov: probably remove that, and operate only on 'pPresentResourceDataAdditional'
    bool isEnhancedHighresEnabled;
    bool generateThumbnail;                                     // This flag enables thumbnail image (regular screenshot, taken as a part of multicapture) generation for super resolution 
    float enhancedHighresCoefficient;
    const AnselResourceData * pPresentResourceData;
    const AnselResourceData * pPresentResourceDataAdditional;   // used only to capture LDR and HDR in the same frame
    ID3D11Texture2D * depthTexture = nullptr;                   // For capturing depth buffer as a layer in psd files
    ID3D11Texture2D * hudlessTexture = nullptr;                 // For capturing hudless buffer as a layer in psd files
    ID3D11Texture2D * colorTexture = nullptr;                   // For capturing color buffer as a layer in psd files
    uint32_t width;
    uint32_t height;
};

struct AnselSDKUIParameters
{
    uint32_t highresMultiplier;
    float sphericalQualityFov;
    float fovSliderValue;
    float rollDegrees;
    float eyeSeparation;
    float cameraSpeedMultiplier;
    bool isCameraDragActive;
    bool isFOVChanged;
    bool isRollChanged;
    bool isHighQualityEnabled;
    AnselUIBase * uiInterface;
};

struct AnselSDKMiscParameters
{
    double dtSeconds;
    bool useHybridController;
};

struct AnselSDKUpdateParameters
{
    float lwrrentHorizontalFov;
    float roll;
    ShotType shotToTake;
    bool needsFOVupdate;
    bool needToDisableSnapButton;
    bool needToFinalizeDirector;
    bool makeScreenshot;
    bool screenshotTaken;
    bool processingCompleted;
    bool isResetRollAvailable;
    std::wstring processedAbsPath;
    AnselUIBase::MessageType captureStatus;
};

// CW-specific
struct AnselSDKStateTelemetry
{
    bool usedGamepadForCameraDuringTheSession;
};

struct AnselUserControlDesc
{
    struct LocalizedLabel
    {
        std::string lang; //'en-US', 'ru-RU', etc
        std::string labelUtf8;
    };

    ansel::UserControlInfo info;
    std::vector<LocalizedLabel> labels;
};

class AnselSDKState: public input::InputEventsConsumerInterface
{
public:
    AnselSDKState();

    const ansel::Configuration& getConfiguration() const;
    const ansel::SessionConfiguration& getSessionConfiguration() const;

    typedef void(*SessionFunc)(void* userData);
    typedef void (__cdecl * BufferFinishedCallback)(void * userData, ansel::BufferType bufferType, uint64_t threadId);

    bool isSDKDetectedAndSessionActive();

    bool startSession(uint32_t width, uint32_t height, bool useHybridController);
    void stopClientSession();   // This function only informs the SDK client that session should be deactivated, and restores the cam controler
    void stopSession();         // This function triggers stopClientSession() and also cleans up resources

    enum class DetectionStatus
    {
        kSUCCESS = 0,
        kDLL_NOT_FOUND = 1, // kDLL_NOT_FOUND is not a fatal status for Ansel, but means that Ansel NoSDK must be used
        kDRIVER_API_MISMATCH = 2,

        kNUM_ENTRIES
    };

    HMODULE AnselSDKState::findAnselSDKModule();

    void setSessionData(SessionFunc sessionStart, SessionFunc sessionStop, void* sessionUserData);

    DetectionStatus detectAndInitializeAnselSDK(const std::wstring& installationPath, 
        const std::wstring& intermediatePath, const std::wstring& snapshotPath,
        SessionFunc sessionStart, SessionFunc sessionStop, BufferFinishedCallback bufferFinished, void* sessionUserData);

    bool isConfigured() const;
    bool isDetected() const;
    bool isSessionActive() const;
    bool isHdrHintsSupported() const;
    ansel::Camera getDisplayCamera() const;
    ansel::Camera getOriginalCamera() const;
    bool isCameraChanged() const;
    bool isCameraInitialized() const;

    void abortCapture();
    void resetCaptureState();

    int getCaptureState() const;
    int getCaptureLatency() const;
    int getSettleLatency() const;
    int getProgressLwrrent() const;
    int getProgressTotal() const;

    uint32_t getAnselSdkVersionMajor() const;
    uint32_t getAnselSdkVersionMinor() const;
    uint32_t getAnselSdkVersionCommit() const;

    const std::string& getTitleForTagging() const { return m_titleForTaggingUtf8; }
    const std::wstring& getTitleForFileNaming() const { return m_titleForFileNaming; }
    const std::wstring& getDrsProfileName() const { return m_drsProfileName; }
    const std::wstring& getDrsAppName() const { return m_drsAppName; }

    void setCaptureState(int captureState);
    void setCaptureLatency(int captureLatency);
    void setSettleLatency(int settleLatency);
    void setProgressLwrrent(int progressLwrrent);
    void setProgressTotal(int progressTotal);

    bool isBufferBindHintActive(ansel::BufferType bufferType, uint64_t& threadId, ansel::HintType& hintType) const;
    bool isBufferFinishedHintActive(ansel::BufferType bufferType, uint64_t& threadId) const;
    void clearBufferBindHint(ansel::BufferType bufferType);
    void clearBufferFinishedHint(ansel::BufferType bufferType);

    // a group of functions related to Ansel SDK user controls
    bool isUserControlUpdated() const;
    void clearUserControlUpdated() const;
    std::vector<AnselUserControlDesc> getUserControlDescriptions() const;
    void userControlChanged(uint32_t id, const void* value) const;

    void updateTileUV(float * tlU, float * tlV, float * brU, float * brV);
    AnselSDKUpdateParameters update(input::InputState * inputCapture, bool isCameraInteractive, ErrorManager & errorManager, const AnselSDKCaptureParameters & mutableParameters, const AnselSDKUIParameters& uiParameters, const AnselSDKMiscParameters & miscParameters, ShotSaver* ss);

    anselutils::CameraControllerFree* getCameraControllerFree();
    void getTelemetryData(AnselSDKStateTelemetry& telemetry) const;
    
    void initTitleAndDrsNames();

    bool isLwrrentShotCapture();

    ansel::UserControlStatus getUserControlValue(uint32_t userControlId, void* value)       { if (m_getUserControlValue) return m_getUserControlValue(userControlId, value); else return ansel::kUserControlIlwalidCallback; }
    ansel::UserControlStatus setUserControlValue(uint32_t userControlId, const void* value) { if (m_setUserControlValue) return m_setUserControlValue(userControlId, value); else return ansel::kUserControlIlwalidCallback; }

private:
    void stopCapture();
    void setCameraControllerRelativeIfNeeded();
    void setNextCameraFromSequence();
    bool processDirectorSequence();
    bool processDirectorLatencies(bool& needToMakeLwrrentShot, bool& needToFinalizeDirector, bool& needsToolsProcess, std::wstring& filePath, AnselSDKUpdateParameters& outputParameters);
    void processInput(input::InputState * inputCapture, bool isCameraInteractive, const AnselSDKUIParameters & uiParameters, const AnselSDKMiscParameters & miscParameters, ErrorManager & errorManager);
    void handleCaptureTaskStartup(AnselSDKUpdateParameters& outputParameters, ErrorManager & errorManager, const AnselSDKCaptureParameters & captureParameters, const AnselSDKUIParameters& parameters);
    void startProcessingDarkroomJob(const std::wstring& path, std::wstring& outputFilename, ErrorManager & errorManager, const AnselSDKCaptureParameters & captureParameters, const AnselSDKUIParameters & uiParameters);
    void handleDarkroomJobFinish(ErrorManager & errorManager, AnselSDKUpdateParameters & outputParameters, const AnselSDKUIParameters & uiParameters, bool keepIntermediateShots);
    darkroom::ShotDescription prepareShotDescription(const AnselSDKCaptureParameters& captureParameters, const AnselSDKUIParameters& uiParameters, bool& allowColwertToJPG);

    virtual void onInputEvent(const input::InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
        const input::MomentaryMouseState& mouseSt,
        const input::MomentaryGamepadState& gpadSt,
        const input::FolwsChecker& folwsChecker,
        const input::MouseTrapper& mouseTrapper) override;

    void callQualityCallback(bool isHighQuality);

    bool m_DLLfound = false;
    bool m_needsFOVupdate;
    bool m_needsDarkroomProcInit;
    bool m_allowColwertToJPG; //determined based on max dim
    bool m_hdrHintsSupported;
    bool m_isCameraMoving = false;

    int m_progressTotal;
    int m_progressLwrrent;

    int m_cameraSettleLatency;  // settle latency counter for camera setup
    int m_fileSaveSettleLatency;    // settle latency counter for screenshot capture
    int m_captureState;
    int m_captureOffsetWait;

    int m_cfgShotSettleLatency;
    int m_cfgShotCaptureLatency;
    float m_cfgMetersInUnit;

    bool m_highQualitySetting = false;

    float m_camFOV;

    uint64_t m_anselSdkVersion;

    std::unique_ptr<darkroom::CameraDirector> m_director;
    anselutils::CameraController* m_anselSavedController;
    anselutils::CameraControllerRelative* m_anselCameraControllerRelative;
    anselutils::CameraControllerFree* m_anselCameraControllerFree;
    ansel::Camera m_savedCamera;

    ansel::Configuration* m_configuration;
    std::unique_ptr<char[]> m_configurationStorage;
    ansel::SessionConfiguration* m_sessionConfiguration;
    std::unique_ptr<char[]> m_sessionConfigurationStorage;

    std::vector<std::wstring> m_messageParams;

    std::vector<DirectorProcJob> m_directorJobs;

    std::wstring m_drsProfileName;
    std::wstring m_drsAppName;
    std::string  m_titleForTaggingUtf8;
    std::wstring m_titleForFileNaming;

    //Telemetry - please don;t use except for stats { 

    bool m_usedGamepadForCameraDuringTheSession;
    //}

    // defined in AnselSDK too

    typedef void(__cdecl *PFNCWINITIALIZECONFIGURATIONFUNC) (ansel::Configuration& cfg);
    typedef void(__cdecl *PFNCWINITIALIZESESSIONCONFIGURATIONFUNC) (ansel::SessionConfiguration& cfg);
    typedef void(__cdecl *PFNCWGETCONFIGURATIONFUNC) (ansel::Configuration& cfg);
    typedef void(__cdecl *PFNCWGETCONFIGURATIONFUNC) (ansel::Configuration& cfg);
    typedef uint32_t(__cdecl *PFNCWGETCONFIGURATIONSIZE) ();
    typedef uint32_t(__cdecl *PFNCWGETSESSIONCONFIGURATIONSIZE) ();
    typedef void(__cdecl *PFNCWSETSESSIONFUNCTIONS) (SessionFunc start, SessionFunc stop, void* userData);
    typedef void(__cdecl *PFNCWSETBUFFERFINISHEDCALLBACK) (BufferFinishedCallback bufferFinishedCallback);
    typedef void(__cdecl *PFNCWSETUPDATECAMERAFUNC) (decltype(ansel::updateCamera)* updateCameraFunc);
    typedef bool(__cdecl *PFNBUFFERBINDHINTFUNC)(ansel::BufferType, uint64_t&, ansel::HintType&);
    typedef bool(__cdecl *PFNBUFFERFINISHEDHINTFUNC)(ansel::BufferType, uint64_t&);
    typedef bool(__cdecl *PFNBUFFERBINDHINTFUNC13)(uint64_t&, ansel::HintType&);
    typedef bool(__cdecl *PFNBUFFERFINISHEDHINTFUNC13)(uint64_t&);
    typedef void(__cdecl *PFNCLEARHINTFUNC)(ansel::BufferType);
    typedef uint64_t(__cdecl *PFNGETVERSIONFUNC)();
    typedef void(__cdecl *PFLWOIDFUNC)();
    typedef bool(__cdecl *PFNBOOLFUNC)();
    typedef ansel::UserControlStatus(__cdecl *PFNADDUSERCONTROLFUNC)(const ansel::UserControlDesc& desc);
    typedef ansel::UserControlStatus(__cdecl *PFNSETUSERCONTROLLABELLOCALIZATIONFUNC)(uint32_t userControlId, const char* lang, const char* labelUtf8);
    typedef ansel::UserControlStatus(__cdecl *PFNREMOVEUSERCONTROLFUNC)(uint32_t userControlId);
    typedef ansel::UserControlStatus(__cdecl *PFNGETUSERCONTROLVALUEFUNC)(uint32_t userControlId, void* value);
    typedef ansel::UserControlStatus(__cdecl *PFNSETUSERCONTROLVALUEFUNC)(uint32_t userControlId, const void* value);
    typedef uint32_t(__cdecl *PFNGETUSERCONTROLDESCRIPTIONSIZEFUNC)();
    typedef void(__cdecl *PFNUSERCONTROLVALUECHANGEDFUNC)(uint32_t id, const void* value);
    typedef void(__cdecl *PFNGETUSERCONTROLDESCRIPTIONFUNC)(uint32_t index,
                                                            ansel::UserControlType& controlType, 
                                                            const char**& labels, 
                                                            uint32_t& labelsCount,
                                                            void*& value, 
                                                            uint32_t& userControlId);

    PFNADDUSERCONTROLFUNC m_addUserControl;
    PFNSETUSERCONTROLLABELLOCALIZATIONFUNC m_setUserControlLabelLocalization;
    PFNREMOVEUSERCONTROLFUNC m_removeUserControl;
    PFNGETUSERCONTROLVALUEFUNC m_getUserControlValue;
    PFNSETUSERCONTROLVALUEFUNC m_setUserControlValue;
    PFNGETUSERCONTROLDESCRIPTIONSIZEFUNC m_getUserControlDescriptionsSize;
    PFLWOIDFUNC m_lockUserControlDescriptions;
    PFLWOIDFUNC m_unlockUserControlDescriptions;
    PFLWOIDFUNC m_clearUserControlDescListDirtyFlag;
    PFNBOOLFUNC m_isUserControlDescListDirty;
    PFNUSERCONTROLVALUECHANGEDFUNC m_userControlValueChanged;
    PFNGETUSERCONTROLDESCRIPTIONFUNC m_getUserControlDescription;
    PFNGETVERSIONFUNC m_getVersion;
    PFNCWGETCONFIGURATIONSIZE m_getConfigurationSize;
    PFNCWGETSESSIONCONFIGURATIONSIZE m_getSessionConfigurationSize;
    PFNCWSETSESSIONFUNCTIONS m_setSessionFunctions;
    PFNCWSETBUFFERFINISHEDCALLBACK m_setBufferFinishedCallback;
    PFNCWSETUPDATECAMERAFUNC m_setUpdateCameraFunc;
    PFNCWGETCONFIGURATIONFUNC m_getConfigurationPfn;

    // Hints API
    PFNCLEARHINTFUNC m_clearBufferBindHint;
    PFNCLEARHINTFUNC m_clearBufferFinishedHint;
    PFNBUFFERBINDHINTFUNC m_getBufferBindHintActive;
    PFNBUFFERFINISHEDHINTFUNC m_getBufferFinishedHintActive;
    // Hints API before Ansel SDK 1.4
    PFLWOIDFUNC m_clearBufferBindHint13;
    PFLWOIDFUNC m_clearBufferFinishedHint13;
    PFNBUFFERBINDHINTFUNC13 m_getBufferBindHintActive13;
    PFNBUFFERFINISHEDHINTFUNC13 m_getBufferFinishedHintActive13;
    // Hints API before Ansel SDK 0.18
    PFNBOOLFUNC m_getBufferBindHintActive017;
    PFNBOOLFUNC m_getBufferFinishedHintActive017;

    PFNCWINITIALIZECONFIGURATIONFUNC m_initializeConfiguration;
    PFNCWINITIALIZESESSIONCONFIGURATIONFUNC  m_initializeSessionConfiguration;
};
