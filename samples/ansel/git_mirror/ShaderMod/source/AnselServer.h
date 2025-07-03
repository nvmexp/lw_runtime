#pragma once

#include "D3DCompilerHandler.h"
#include "Ansel.h" // Ansel.h depends on D3DCompilerHandler.h being included before it

// ShaderMod includes
#include "RegistrySettings.h"
#include "AnselSDKState.h"
#if (ENABLE_CONTROL_SDK == 1)
#include "AnselControlSDKState.h"
#endif
#include "MultipassConfigParserError.h"
#include "NetworkDetection.h"
#include "ir/IRCPPHeaders.h"
#include "ShaderModMultiPass.h"
#include "UIBase.h"
#include "darkroom/InternalLimits.h"
#include "sideload/SideloadHandler.h"
#include "DebugRenderer.h"
#include "DenylistParser.h"
#include "LocalizedFilterNames.h"
#include "EffectsInfo.h"
#include "Hash.h"
#include "RenderBuffer.h"
#include "RenderBufferColwert.h"
#include "BufferTestingOptionsFilter.h"

// PSD
#include "PhotoShopUtils.h"

#ifdef ENABLE_STYLETRANSFER
#include "lwda_runtime_api.h"
#include "style_transfer.h"
#endif

#include <d3d11.h>
#include <Windows.h>
#include <stdint.h>
#include <assert.h>
#include <map>
#include <list>
#include <vector>
#include <unordered_set>
#include <memory>

#define DBG_MODDING_PAST_UI_PROTO   0

#define IS_SDK_REQUIRED             1

#define ENABLE_DEBUG_RENDERER       0

class AnselUI;
class AnselIPCMessageBusObserver;

struct GlobalPerfCounters
{
    void update()
    {
        dt = timer.Time();
        elapsedTime += dt * 0.001;
        timer.Start();
    }

    shadermod::Timer    timer;
    double              dt = 0.0;           // ms
    double              elapsedTime = 0.0;
};

struct AnselBufferSnapshot
{
    HCLIENTRESOURCE hLwrrentDSBuffer;
    HCLIENTRESOURCE hSelectedHUDless;
    HCLIENTRESOURCE hSelectedHDR;
};

// data that needs to be associated per-command list in the driver
class AnselCommandList
{
public:
    AnselCommandList() { BufferSnapshots = new std::vector<AnselBufferSnapshot*>(); }
    virtual ~AnselCommandList()
    {
        Reset();
        delete BufferSnapshots;
    }

    void Reset() {
        for (unsigned int i = 0; i < BufferSnapshots->size(); ++i)
        {
            delete BufferSnapshots->at(i);
        }
        BufferSnapshots->clear();
    }
    void AddSnapshot(AnselBufferSnapshot *snapshot) { BufferSnapshots->push_back(snapshot); }

private:
    std::vector<AnselBufferSnapshot*> *BufferSnapshots;
};

class AnselAutoCS
{
    CRITICAL_SECTION * m_pCS;
public:
    AnselAutoCS(CRITICAL_SECTION * pCS)
        : m_pCS(pCS)
    {
        EnterCriticalSection(pCS);
    }

    ~AnselAutoCS()
    {
        LeaveCriticalSection(m_pCS);
    }
};

class AnselServer: public shadermod::MultiPassProcessor, public ShotSaver
{
public:

    AnselD3D11BufferInterface m_d3d11Interface;
    AnselD3D12BufferInterface m_d3d12Interface;

    AnselBufferInterface * m_lwrrentBufferInterface = nullptr;

    AnselBufferColwerter m_renderBufferColwerter;

    void setupColwerter()
    {
        m_renderBufferColwerter.setRenderer(m_d3dDevice, m_immediateContext, &m_d3dCompiler);
        m_renderBufferColwerter.init();
    }

    void initBufferInterfaces()
    {
        m_d3d11Interface.setClientData(m_hClient, m_pClientFunctionTable, &m_handleToAnselResource);
        m_d3d11Interface.setServerGAPIData(m_d3dDevice, m_immediateContext);

        m_d3d12Interface.setClientData(m_hClient, m_pClientFunctionTable, &m_handleToAnselResource);
        m_d3d12Interface.setServerGAPIData(m_d3dDevice, m_immediateContext);

        // TODO avoroshilov: implement D3D12 notifications check to change the lwrrentBufferInterface
        //  or even better, reorganize the structure, common parts should have common storage
        // Do not ilwoke buffer interface setup here, as we don't know what API would we use at this point
        //setBufferInterfaceD3D11();
    }

    void setBufferInterfaceD3D11()
    {
        m_lwrrentBufferInterface = &m_d3d11Interface;
        m_bufDB.setBuffersInterfaceIfNeeded(m_lwrrentBufferInterface);
    }
    void setBufferInterfaceD3D12()
    {
        m_lwrrentBufferInterface = &m_d3d12Interface;
        m_bufDB.setBuffersInterfaceIfNeeded(m_lwrrentBufferInterface);

        // DX12 games don't support stats mechanism yet; disable it for now otherwise
        // the Stats resolve step will cause games to have Depth missing
        m_bufDB.Depth().setStatsEn(false);
    }

    bool m_bInitialized;
    bool m_bInitializedOK = true;

    bool m_isAnselActive = true;

    bool m_areMouseButtonsSwapped = false;

    std::wstring m_toolsFolderPath;

    std::wstring m_effectInstallationFolderPath;
    std::wstring m_effectUserFolderPath;
    std::vector<std::wstring> m_effectFoldersAdditional;

    std::wstring m_userStylesFolderPath;
    std::wstring m_intermediateFolderPath;
    std::wstring m_snapshotsFolderPath;
    std::wstring m_installationFolderPath;

    int32_t m_maxHighResResolution;
    int32_t m_maxSphericalResolution;
    float m_eyeSeparation;
    float m_cameraSpeedMultiplier;

    LCID m_forcedLocale = 0;

    bool m_removeBlackTint = false;
    bool m_keepIntermediateShots = false;
    bool m_renderDebugInfo = false;
    bool m_losslessOutputSuperRes = false;
    bool m_losslessOutput360 = false;

    bool m_useHybridController = false;

    bool m_allowNotifications = true;

    bool m_allowTelemetry = true;
    bool m_requireAnselSDK = true;
    bool m_standaloneModding = false;
    bool m_startAnselSDKSession = true;
    bool m_allowFiltersInGame = false;
    bool m_allowDynamicFilterStacking = true;

    double m_nextNetworkActivityModdingCheck = -1.0;
    bool m_networkActivityDetected = false;

    bool m_enableEnhancedHighres = false;
    const float m_enhancedHighresCoeffAggressive = 0.1f;
    const float m_enhancedHighresCoeffMinimal = 1.1f;   // This value is needed because we limit cfg tool UI to 10%-100%, so we need 10% to map at 1.0
    float m_enhancedHighresCoeff = 0.75f;

    USHORT m_toggleHotkeyModCtrl = 0;
    USHORT m_toggleHotkeyModShift = 0;
    USHORT m_toggleHotkeyModAlt = 1;
    USHORT m_toggleAnselHotkey = VK_F2;
    std::wstring m_toggleHotkeyComboText;

    bool m_lightweightShimMode = false;
    bool m_bNotifyRMAnselStop = false;

    HANSELCLIENT m_hClient;
    ClientFunctionTable * m_pClientFunctionTable;
    GlobalPerfCounters m_globalPerfCounters;
    //IDXGIFactory1 * pFactory;
    //IDXGIAdapter * pAdapter;
    //ID3D11Device * pDevice;
    //ID3D11DeviceContext * pContext;

    std::map<HCLIENTRESOURCE, AnselResource *> m_handleToAnselResource;
    std::mutex m_handleToAnselResourceLock;
    AnselEffectState m_passthroughEffect;
    AnselEffectState m_gridEffect;

#if DBG_HARDCODED_EFFECT_BW_ENABLED
    AnselEffectState m_blackAndWhiteEffect;
#endif

    ANSEL_EXEC_DATA *m_pLwrrentExecData = nullptr;

    AnselDeviceStates m_deviceStates;

#ifdef ENABLE_STYLETRANSFER
    std::vector<std::wstring>   m_stylesFilesList;          // Raw filenames
    std::vector<std::wstring>   m_stylesFilesListTrimmed;   // Trimmed filenames (w/o extension)
    std::vector<std::wstring>   m_stylesFilesPaths;         // Full paths to style files
    void populateStylesList();

    std::vector<std::wstring>   m_styleNetworksLabels;
    std::vector<std::wstring>   m_styleNetworksIds;
    std::wstring m_oldStyleTelemetry;
    void populateStyleNetworksList();

    sideload::SideloadHandler m_sideloadHandler;
    std::vector<std::pair<std::vector<float>, std::vector<float>>> m_styleTransferStyles;
    size_t m_styleIndex = 0u;
    bool m_wasStyleTransferFolderShown = false;
    int m_styleSelected;
    bool m_isStyleTransferEnabled = false;
    bool m_allowStyleTransferWhileMovingCamera = false;
    bool m_refreshStyleTransfer = false;
    bool m_refreshStyleTransferAfterCapture = false;

    std::map<std::wstring, std::pair<std::vector<float>, std::vector<float>>> m_styleStatisticsCache;

    bool m_styleTransferHDR = false;

    struct StyleTransferTextureBuffer
    {
        ID3D11Texture2D *       tex = nullptr;
        DXGI_FORMAT             fmt = DXGI_FORMAT_UNKNOWN;
        size_t                  w = 0;
        size_t                  h = 0;
    };

    StyleTransferTextureBuffer m_styleTransferOutputBuffer;
    StyleTransferTextureBuffer m_styleTransferOutputHDRStorageBuffer;

    lwdaDeviceProp          m_lwdaDeviceProps;
    int getStyleIndexByName(const std::wstring& filterName) const;
    std::chrono::time_point<std::chrono::steady_clock> m_downloadStarttime;
    std::wstring generateRestyleLibraryName(const std::wstring& path, const std::wstring& baseName, const uint32_t major, const uint32_t minor, const std::wstring& versionString, bool debug = false);
    std::pair<std::vector<float>, std::vector<float>> loadStyleTransferCache(const std::wstring& styleFilename, const std::wstring& stylePath);
#endif

    ErrorManager m_errorManager;

    std::wstring m_appName;
    std::wstring m_deviceName;

#if DBG_HARDCODED_EFFECT_BW_ENABLED
    bool m_bEnableBlackAndWhite;
    bool m_bNextFrameEnableBlackAndWhite;
#endif

    bool m_bWasAnselDeactivated = true;

    bool m_enableDepthExtractionChecks = true;
    bool m_depthBufferUsed = false;
    bool m_bRenderDepthAsRGB;
    bool m_enableHDRExtractionChecks = true;
    bool m_hdrBufferUsedByFilter = false;
    bool m_hdrBufferUsedByShot = false;
    bool isHDRBufferUsed() { return (m_hdrBufferUsedByFilter || m_hdrBufferUsedByShot); }

    bool m_hdrBufferAvailable = false;
    bool m_enableHUDlessExtractionChecks = true;
    bool m_hudlessBufferUsed = false;
    bool m_enableFinalColorExtractionChecks = true;
    bool m_bRunShaderMod;
    bool m_bShaderModInitialized = false;

    bool m_makeScreenshotNextFrame = false;
    bool m_makeScreenshotNextFrameValue = false;
    ShotType m_shotToTakeNextFrame = ShotType::kNone;

    PhotoShopUtils m_psUtil;

    // by default we hide our mouse pointer when defolwsed to avoid double mouse
    bool m_showMouseWhileDefolwsed = false;

    // Forcing fade state setup on the first frame
    bool m_bNextFrameEnableFade = false;
    bool m_bNextFrameDisableFade = false;

    bool m_enableAnselModding = false;
    bool m_moddingDepthBufferAllowed = true;
    bool m_moddingHDRBufferAllowed = true;
    bool m_moddingHUDlessBufferAllowed = true;

    bool m_bNextFrameForceEnableAnsel = false;
    bool m_bNextFrameForceDisableAnsel = false;

    bool m_bNextFrameRenderDepthAsRGB;
    bool m_bNextFrameRunShaderMod;

    bool m_bNextFramePrevEffect;
    bool m_bNextFrameNextEffect;
    bool m_bNextFrameRebuildYAML;
    bool m_bNextFrameRefreshEffectStack = false;

    bool m_stackRebuildRequired = false;

    PFND3DCOMPILEFUNC m_D3DCompileFunc;

    ShotType m_shotToTake = ShotType::kNone;
    bool m_makeScreenshotHDR = false;
    bool m_makeScreenshotHDRJXR = false;
    bool m_makeScreenshot = false, m_makeScreenshotWithUI = false;

    int m_prevWidth = -1, m_prevHeight = -1;
    size_t forward_vram_estimation = 0;

    bool m_isClientEnabled = false;     // This is basically an equivalent of UI being enabled (set on session start, removed on session stop)
    bool m_isClientActive = false;      // This is an equivalent of UI being in active state, i.e. including pre-start and post-stop (set on session pre-start, removed on session post-stop)

    enum class ModdingMode
    {
        kDisabled = 0,
        kEnabled = 1,
        kRestrictedEffects = 2,
    };
    enum class ModdingBuffersUse : uint32_t
    {
        kDisallowDepth =        0b00000000000000000000000000000010,
        kDisallowHUDless =      0b00000000000000000000000000000100,
        kDisallowHDR =          0b00000000000000000000000000001000,

        kAllowAll = 0,
        kDisallowAllExtra = kDisallowDepth | kDisallowHUDless | kDisallowHDR
    };

    struct ModdingSettings
    {
        ModdingMode mode;
        uint32_t restrictedEffectSetId;
        ModdingBuffersUse bufferUse;
    };

    enum class ModdingRestrictedSetID : uint32_t
    {
        kPrepackaged = 0,
        kMPApproved = 1,

        kNUM_ENTRIES,

        kEmpty = (uint32_t)-1
    };

    ModdingSettings m_moddingSinglePlayer;
    ModdingSettings m_moddingMultiPlayer;
    uint32_t m_freeStyleModeValueVerif;

    void parseModdingModes(uint32_t modeFlags, ModdingSettings * moddingSP, ModdingSettings * moddingMP);

    enum class PrepackagedEffects
    {
        // Yaml Filters
        kAdjustments_yaml,
        kBlacknWhite_yaml,
        kColor_yaml,
        kColorblind_yaml,
        kDOF_yaml,
        kDetails_yaml,
        kLetterbox_yaml,
        kNightMode_yaml,
        kOldFilm_yaml,
        kPainterly_yaml,
        kRemoveHud_yaml,
        kSharpen_yaml,
        kSpecialFX_yaml,
        kSplitscreen_yaml,
        kTiltShift_yaml,
        kVignette_yaml,
        kWatercolor_yaml,

        // ReShade Filters
        k3DFX_fx,
        kASCII_fx,
        kAdaptiveSharpen_fx,
        kAmbientLight_fx,
        kBloom_fx,
        kBorder_fx,
        kCRT_fx,
        kCartoon_fx,
        kChromaticAberration_fx,
        kClarity_fx,
        kColourfulness_fx,
        kLwrves_fx,
        kDaltonize_fx,
        kDeband_fx,
        kDenoise_fx,
        kDenoiseKNN_fx,
        kDenoiseNLM_fx,
        kFXAA_fx,
        kFakeMotionBlur_fx,
        kGaussianBlur_fx,
        kGlitch_fx,
        kHQ4X_fx,
        kMonochrome_fx,
        kNightVision_fx,
        kPPFX_Godrays_fx,
        kPrism_fx,
        kSMAA_fx,
        kTest_Reshade_Hashed_fx,
        kTiltShift_fx,
        kTriDither_fx,
        kVignette_fx,

        // Special shader, used in the system tests
        kTesting_yaml,

        kNUM_ENTRIES,
    };

    static const std::map<std::string, PrepackagedEffects> s_effectStringToEnum;

    std::vector<std::map<Hash::Data, PrepackagedEffects>> m_moddingRestrictedSets;

    class ModdingEffectHashDB
    {
    public:
        using EffectsMap = std::map<PrepackagedEffects, std::set<Hash::Effects>>;

        static bool CompareMainHash(const Hash::Data& hash);

        static const Hash::Data GetGeneratedMainHash();

        static bool GetGeneratedShiftedHashSet(PrepackagedEffects effIdx, std::set<Hash::Effects>& retHash);

        static const Hash::Data GenerateMainHash(const ModdingEffectHashDB::EffectsMap &refEffects);

    private:
        static const EffectsMap GenerateShiftedHashSet();

        static const EffectsMap m_effHashes;
        static const EffectsMap m_effShiftedHashes;

        static const Hash::Data m_mainHash;
        static const Hash::Data m_generatedMainHash;
    };

    static const ModdingEffectHashDB m_moddingEffectHashDB;

    uint32_t m_moddingHashVerif0PrepStep = 0;

    EffectsInfo m_effectsInfo;

    void resizeEffectsInfo(size_t newEffectsNum);

    bool m_hadNoActiveEffects = true;// Defaults is true because at startup, there were previously no effects applied.

    std::vector<std::wstring> m_storageEffectFilenames;
    std::vector<std::wstring> m_storageEffectRootFolders;

    bool m_welcomeNotificationLogged = false;

    bool m_wasModdingAllowed = false;
    bool m_wasModdingAllowedGlobally = false;
    bool m_shownNetworkActivityMsg = false;
    bool m_enableWelcomeNotification = false;
    bool m_performModsEnabledCheck = false;
    bool m_checkTraficLocal = true;
    bool m_modEnableFeatureCheck = true;
    bool m_sendAnselReady = false;
    std::chrono::time_point<std::chrono::steady_clock> m_modsAvailableQueryTimepoint;
    std::map<std::string, std::wstring> m_settingsAsStrings;

    std::vector<std::wstring> m_displayMessageStorage;

    std::vector<AnselUIBase *> m_anselClientsList;

    bool m_specialClientActive = false;
    bool m_UiUpdateSdkStatusNeeded = true;
    AnselUIBase * m_activeControlClient = nullptr;
    AnselUIBase * m_UI;
    AnselSDKState m_anselSDK;
#if (ENABLE_CONTROL_SDK == 1)
    AnselControlSDKState m_anselControlSDK;
    uint32_t m_anselControlSDKDetectAndInitializeAttempts = 0;
#endif

    HINSTANCE m_hDLLInstance;

    AnselServer(HINSTANCE hDLLModule);
    ~AnselServer();

    void changeEffectState(bool enableShaderMod);

    HRESULT init(HANSELCLIENT hClient, ClientFunctionTable * pFunctionTable, LARGE_INTEGER adapterLUID);
    HRESULT initCanFail(HANSELCLIENT hClient, ClientFunctionTable * pFunctionTable, LARGE_INTEGER adapterLUID);
    bool isInitialized() { return m_bInitialized; }
    HRESULT createSharedResource(DWORD width,
                                DWORD height,
                                DWORD sampleCount,
                                DWORD sampleQuality,
                                DWORD format,
                                HANDLE * pHandle,
                                void * pServerPrivateData);
    HRESULT finalizeFrame(HCLIENTRESOURCE hPresentResource, DWORD subResIndex);
    HRESULT exelwtePostProcessing(HCLIENTRESOURCE hPresentResource, DWORD subResIndex);
    void deactivateAnsel(bool doCleanup = true);
    void enterLightweightMode();
    void exitLightweightMode();

    //Notifies the Frame Rate Monitor that Ansel has completed all work in one frame.
    // deactivatingAnsel should be true when Ansel is transitioning into lightweight or passthrough mode
    // notifyFrameComplete(true) must be called when Ansel transitions into passthrough mode without destroying its device.
    HRESULT notifyFrameComplete(bool deactivatingAnsel);

    void disconnectIpc();
    HRESULT destroy();

    HRESULT eraseClientResourceHandleIfFound(HCLIENTRESOURCE hClientResource);

    HRESULT bufferFinished(ansel::BufferType bufferType, uint64_t threadId);

    HCLIENTRESOURCE m_lastBoundDepthResource = nullptr;
    bool m_wasDepthHintUsed = false;

    HCLIENTRESOURCE m_lastBoundColorResource = nullptr;

    bool m_wasHDRHintUsed = false;
    HRESULT checkHDRHints(HCLIENTRESOURCE hdrResource, const AnselClientResourceInfo & resourceInfo);

    bool m_wasHUDlessHintUsed = false;
    HRESULT checkHUDlessHints(HCLIENTRESOURCE hudlessResource, const AnselClientResourceInfo & resourceInfo);

    bool m_wasFinalColorHintUsed = false;
    HRESULT checkFinalColorHints(HCLIENTRESOURCE colorResource, const AnselClientResourceInfo & resourceInfo);

    HRESULT notifyDraw();
    HRESULT notifyDepthStencilCreate(HCLIENTRESOURCE hClientResource);
    HRESULT notifyDepthStencilDestroy(HCLIENTRESOURCE hClientResource);
    HRESULT notifyDepthStencilBind(HCLIENTRESOURCE hClientResource);
    HRESULT notifyRenderTargetBind(HCLIENTRESOURCE* phClientResource, DWORD dwNumRTs);
    HRESULT notifyUnorderedAccessBind(DWORD startOffset, DWORD count, HCLIENTRESOURCE* phClientResource);
    HRESULT notifyClientResourceDestroy(HCLIENTRESOURCE hClientResource);
    HRESULT notifyDepthStencilClear(HCLIENTRESOURCE hClientResource);
    HRESULT notifyRenderTargetClear(HCLIENTRESOURCE hClientResource);

    HRESULT notifyClientMode(DWORD clientMode);

    HRESULT updateGPUMask(DWORD activeGPUMask);
    std::wstring getD3dCompilerFullPath() const;

    struct AnselBuffers
    {
        AnselResource * pPresentResourceData;
        AnselResource * pHDRResourceData;
        AnselResource * pDepthResourceData;
        AnselResource * pHUDlessResourceData;
    };
    HRESULT acquireSharedResources(HCLIENTRESOURCE hPresentResource, DWORD subResIndex, AnselBuffers * anselBuffers);
    HRESULT releaseSharedResources(HCLIENTRESOURCE hPresentResource, DWORD subResIndex, AnselBuffers * anselBuffers);
    HRESULT acquireSharedResources12(HCLIENTRESOURCE hPresentResource, DWORD subResIndex, AnselBuffers * anselBuffers);
    HRESULT releaseSharedResources12(HCLIENTRESOURCE hPresentResource, DWORD subResIndex, AnselBuffers * anselBuffers);

    HRESULT notifyCmdListCreate12(HCMDLIST hCmdList, HANSELCMDLIST *phAnselCmdList);
    HRESULT notifyCmdListDestroy12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList);
    HRESULT notifyCmdListReset12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList);
    HRESULT notifyCmdListClose12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList);
    HRESULT notifySetRenderTargetBake12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, void ** ppServerData);
    HRESULT notifySetRenderTargetBakeWithDeviceStates12(const AnselDeviceStates DeviceStates, HANSELCMDLIST hAnselCmdList, void ** ppServerdata);
    HRESULT notifySetRenderTargetExec12(ANSEL_EXEC_DATA *pAnselExecData);
    HRESULT notifyPresentBake12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, void ** ppServerData);
    HRESULT exelwtePostProcessing12(ANSEL_EXEC_DATA *pAnselExecData, HCLIENTRESOURCE hPresentResource, DWORD subResIndex);
    HRESULT exelwtePostProcessing12CanFail(ANSEL_EXEC_DATA *pAnselExecData, HCLIENTRESOURCE hPresentResource, DWORD subResIndex);

    HRESULT notifyDepthStencilClearBake12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, HCLIENTRESOURCE hDepthStencil, void ** pServerData);
    HRESULT notifyDepthStencilClearExec12(ANSEL_EXEC_DATA *pExelwtionContext);

    HRESULT setConfig(AnselConfig * pConfig);
    HRESULT notifyHotkey(DWORD vkey);

    HRESULT saveShot(const AnselSDKCaptureParameters &captureParams, bool forceRGBA8Colwersion, const std::wstring& shotName, bool useAdditionalResource = false) override;
    void debugRenderFrameNumber();

    void abortCapture();

    void reportFatalError(const char* filename, int lineNumber, FatalErrorCode code, const char* format, va_list args) const;
    void reportNonFatalError(const char* filename, int lineNumber, unsigned int code, const char* format, va_list args) const;

    void reportFatalError(const char* filename, int lineNumber, FatalErrorCode code, const char* format, ...) const;
    void reportNonFatalError(const char* filename, int lineNumber, unsigned int code, const char* format, ...) const;

    // ipc related functions
    std::vector<std::wstring> getFilterList() const;
    uint32_t getScreenWidth() const;
    uint32_t getScreenHeight() const;
    uint32_t getMaximumHighresResolution() const;

    bool m_ipcValueInitialized = false;
    int m_WAR_IPCmodeEnabled = 0;
    void setIPCModeEnabled(int state) { m_WAR_IPCmodeEnabled = state; }
    int getIPCModeEnabled() const { return m_WAR_IPCmodeEnabled; }

    int getFilterIndexById(const std::wstring& filterName) const;
    void effectFilterIdAndName(const std::wstring & effectPath, const std::wstring & effectFilename, LANGID langId, std::wstring & filterId, std::wstring & filterDisplayName) const;
    void getAllFileNamesWithinFolders(const std::vector<std::wstring> & exts, std::vector<std::wstring> * outFilenames, std::vector<std::wstring> * outFolders) const;
    bool getEffectDescription(const shadermod::MultiPassEffect* eff, LANGID langID, AnselUIBase::EffectPropertiesDescription * effectDesc) const;
    void populateEffectsList(uint32_t restrictedSetID);

    // This sends UI signal to switch effect to "None", and the actual effect will be automatically unloaded next frame
    void AnselServer::unloadEffectUIClient(size_t effIdx);
    // This function modifies internal state and also sends the UI signal, use with caution
    void AnselServer::unloadEffectInternal(size_t effIdx);
    // This function is similar to previous, plus destroying the effect on internal stack
    void AnselServer::destroyEffectInternal(size_t effIdx);
    void applyEffectChange(AnselUIBase::EffectChange& effectChange, shadermod::MultiPassEffect* eff);
    bool applyEffectChanges();
    shadermod::MultipassConfigParserError loadEffectOnStack(size_t effIdx, size_t internalEffIdx, bool replace, const std::set<Hash::Effects>* pExpectedHashSet, bool compareHashes);

    //Telemetry {
    struct AnselStateForTelemetry
    {
        std::string exeName;
        std::string rawProfileName;
        std::string drsAppName;

        unsigned int width = 0;
        unsigned int height = 0;
        unsigned int colorBufferFormat = 0;
        unsigned int depthBufferFormat = 0;

        UISpecificTelemetryData uiSpecificData;
        float fov360 = 0.0f; //obsolete

        bool specialFxEnabled = false;
        std::string effectNames;

        float dispCam_position_x = 0.0f;
        float dispCam_position_y = 0.0f;
        float dispCam_position_z = 0.0f;
        float dispCam_rotation_x = 0.0f;
        float dispCam_rotation_y = 0.0f;
        float dispCam_rotation_z = 0.0f;
        float dispCam_rotation_w = 0.0f;
        float origCam_position_x = 0.0f;
        float origCam_position_y = 0.0f;
        float origCam_position_z = 0.0f;
        float origCam_rotation_x = 0.0f;
        float origCam_rotation_y = 0.0f;
        float origCam_rotation_z = 0.0f;
        float origCam_rotation_w = 0.0f;

        struct UserVarData
        {
            shadermod::ir::UserConstDataType type = shadermod::ir::UserConstDataType::NUM_ENTRIES;
            std::string value;
            std::string name;
        } top5Var[5];

        std::string packedUserConstants;

        bool isInIPCMode = false;
        bool usedGamepadForCameraDuringTheSession = false;
        double sessionDuration = 0.0;
        double captureDuration = 0.0;

        DWORD gamepadProductId = 0;
        DWORD gamepadVendorId = 0;
        DWORD gamepadVersionNumber = 0;
        input::GamepadDevice::EGamepadDevice gamepadDeviceType = input::GamepadDevice::EGamepadDevice::kUnknown;
        bool highresEnhancementEnabled = false;
        uint32_t anselSDKMajor = 0u;
        uint32_t anselSDKMinor = 0u;
        uint32_t anselSDKCommit = 0u;
    };

    struct ErrorDescForTelemetry
    {
        std::string errorMessage;
        unsigned int errorCode;
        std::string filename;
        unsigned int lineNumber;
    };

    struct
    {
        std::vector<int> oldStackIdx;   // stackIdx of the effect before rearrangement (should be tracked on the UI client side)

        std::vector<shadermod::ir::TypelessVariable> values;
        std::vector<std::string> names; //names of all user constants should be unique in the yaml file. UIDs are only uniqe for the time the constant exists in the contact manager
        std::vector<int> offsets;
        std::vector<std::wstring> effectStack;
    } userConstantStorage;

    AnselStateForTelemetry makeStateSnapshotforTelemetryBody() const;
    HRESULT makeStateSnapshotforTelemetry(AnselStateForTelemetry & state) const;
    double getSessionDuration() const;
    double getCaptureDuration() const;

    bool isTelemetryAvailable() const;
    void destroyTelemetry() const;


    void sendTelemetryStyleTransferDownloadStartedEvent(const std::string& url, const std::string& version, uint32_t computeCapMajor, uint32_t computeCapMinor) const;
    void sendTelemetryStyleTransferDownloadFinishedEvent(uint32_t secondsSpent, int32_t status) const;
    void sendTelemetryStyleTransferStatusEvent(const uint32_t reason, const std::string& comment = std::string()) const;
    void sendTelemetryMakeSnapshotEvent(const AnselStateForTelemetry& state) const;
    void sendTelemetryAbortCaptureEvent(const AnselStateForTelemetry& state) const;
    void sendTelemetryCloseUIEvent(const AnselStateForTelemetry& state) const;
    void sendTelemetryLwCameraErrorShortEvent(const ErrorDescForTelemetry& errDesc, bool isFatal = true) const;
    void sendTelemetryLwCameraErrorEvent(const AnselStateForTelemetry& state, int captureState, const ErrorDescForTelemetry& errDesc, bool isFatal = true) const;
    void sendTelemetryEffectCompilationErrorEvent(const AnselStateForTelemetry& state, int captureState, const ErrorDescForTelemetry& errDesc) const;

    RegistrySettings & getRegistrySettings() { return m_registrySettings; }

private:
    // Finalize Frame Steps
    void enableControlSdk();

    // Helpful functions for debugging/testing
    void debugForceRenderHudless(HCLIENTRESOURCE hPresentResource, DWORD subResIndex);

    void SaveHashes();
    void DenylistBufferExtractionType(ansel::BufferType bufferType);
    std::unordered_set<ansel::BufferType> m_denylistedBufferExtractionTypes;
    AnselDenylist m_denylist;

    void handleDrsGameSettings();

    shadermod::LocalizedEffectNamesParser m_localizedEffectNamesParser;

// style transfer
#ifdef ENABLE_STYLETRANSFER
    decltype(restyleUpdateStyle)* restyleUpdateStyleFunc = nullptr;
    decltype(restyleIsInitialized)* restyleIsInitializedFunc = nullptr;
    decltype(restyleEstimateVRamUsage)* restyleEstimateVRamUsageFunc = nullptr;
    decltype(restyleInitializeWithStyle)* restyleInitializeWithStyleFunc = nullptr;
    decltype(restyleInitializeWithStyleStatistics)* restyleInitializeWithStyleStatisticsFunc = nullptr;
    decltype(restyleCalcAdainMoments)* restyleCalcAdainMomentsFunc = nullptr;
    decltype(restyleForward)* restyleForwardFunc = nullptr;
    decltype(restyleForwardHDR)* restyleForwardHDRFunc = nullptr;
    decltype(restyleDeinitialize)* restyleDeinitializeFunc = nullptr;
    decltype(restyleResizeUsed)* restyleResizeUsedFunc = nullptr;
    decltype(restyleGetVersion)* restyleGetVersionFunc = nullptr;
#endif

    mutable unsigned int m_telemetryRetryAttempts = 0;
    mutable bool m_telemetryInitialized = false;

    // } Telemetry

#if ENABLE_NETWORK_HOOKS == 1
    NetworkActivityDetector m_networkDetector;
#endif
    RegistrySettings m_registrySettings;

    double m_sessionStartTime; //stats
    double m_captureStartTime; //stats

    int m_sessionFrameCount = 0;

    AnselUIBase::Status startSession(bool isModdingAllowed, uint32_t effectsRestrictedSetIDPhoto, uint32_t effectsRestrictedSetIDModding);
    void stopSession();
    HRESULT createPassthroughEffect(AnselEffectState * out);
    HRESULT createGridEffect(AnselEffectState * out);
    HRESULT createBlackAndWhiteEffect(AnselEffectState * out);
    HRESULT createDepthRenderEffect(AnselEffectState * out);
    HRESULT createDepthRenderRGBEffect(AnselEffectState * pOut);
    ID3D11Texture2D* createFullscreenTexture(DXGI_FORMAT fmt, D3D11_USAGE usage, UINT bindFlags, UINT cpuAccess);

    AnselResource * lookupAnselResource(HCLIENTRESOURCE hClientResource);
    void releaseAnselResource(AnselResource * pAnselResource);
    void releaseEffectState(AnselEffectState * pEffectState);

    void initRegistryDependentPathsAndOptions();

    std::vector<std::wstring> m_fxExtensions;
    std::map<std::wstring, std::wstring> m_fxExtensionToolMap;
    HRESULT parseEffectCompilersList(std::wstring filename);

    HRESULT dumpFilterStack();

    void saveBufferFormats(DXGI_FORMAT clr, DXGI_FORMAT depth)
    {
        m_colorBufferFormat = clr, m_depthBufferFormat = depth;
    }

    std::wstring generateActiveEffectsTag() const;
    std::wstring generateSoftwareTag() const;

    int m_toggleDelay = 0;

    AnselBufferDB m_bufDB;

    BufferTestingOptionsFilter m_bufTestingOptionsFilter;

    //this is saved for stats ATM, try not to use
    DXGI_FORMAT             m_colorBufferFormat;
    DXGI_FORMAT             m_depthBufferFormat;


    unsigned int            m_readbackWidth = 0, m_readbackHeight = 0;
    ID3D11Texture2D *       m_readbackTexture = nullptr;
    ID3D11Texture2D *       m_resolvedPrereadbackTexture = nullptr;

    DXGI_FORMAT             m_readbackFormat = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT             m_resolvedPrereadbackFormat = DXGI_FORMAT_UNKNOWN;
    unsigned char *         m_readbackData = nullptr;

    unsigned int            m_readbackTonemappedWidth = 0, m_readbackTonemappedHeight = 0;
    unsigned char *         m_readbackTonemappedData = nullptr;

    // Debug font data
    DebugRenderer m_debugRenderer;

    CRITICAL_SECTION m_csExec;

    HRESULT notifySetRenderTargetBake12Common(const AnselDeviceStates& deviceStates, HANSELCMDLIST hAnselCmdList, void ** ppServerData);
};

