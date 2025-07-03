#pragma once

#include <windows.h>

// Macro to generate per-structure version for use with API.
#define  SP_MAKE_VERSION(typeName, ver) (UINT32)(sizeof(typeName) | ((ver)<<16))

#define SHADOWPLAY_DUMMY_ARG    0xFFFFFFFF
struct CheckStatusArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    UINT32                      allowed;              // [out] Is shadowplay allowed(Enabled && Valid Driver version && other checks)
    
};

struct DirectFlipState
{
    union
    {
        struct
        {
            UINT32 DirectFlipCapable : 1;              // directflip Capable
            UINT32 SharedPrimaryTransitionDone : 3;    // We are transitioning to or away from a shared managed primary allocation
            UINT32 isIndependentFlipExclusive : 1;     // OS using I-flipMode
            UINT32 Reserved : 27;
        };
        UINT32 Value;
    };
};

struct OSCSurfaceInfo
{
    UINT32           numSurfaces;
    UINT32           surfaceSize;
    UINT64           oscSurfaceAddress;
    UINT64           oscControlAddress;
};
   
struct CreateSessionArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    UINT32                      d3dVersion;                  // [in] 
    HANDLE                      hLogoRenderer;  
    UINT32                      rendererVersion;
    UINT32                      isSurround;           // [in] 
    UINT32                      isInBlackList;        // [in] 
    UINT32                      isOptimus;            // [in] 
    wchar_t                     *pszProfileName;
};

struct CreateSessionArgs_V2: CreateSessionArgs
{
    UINT32                      isMSHybrid;           // [in]
};

// shadowPlayFlags masks
// Bit0: 1 - Game, 0 - Non-Game
#define SHADOWPLAY_FLAGS_MASK_IS_GAME           0x00000001
// Bit27: 1 - Explicit SLI codepath enable, 0 - not enabled
#define SHADOWPLAY_FLAGS_EXPLICIT_SLI           0x08000000
// Bit30: 1 - Don't load DX11 shim, 0 - load
#define SHADOWPLAY_FLAGS_DISALLOW_DX11_SHIM     0x40000000
// Bit31: 1 - Don't load DX12 shim, 0 - load
#define SHADOWPLAY_FLAGS_DISALLOW_DX12_SHIM     0x80000000

struct CreateSessionArgs_V3: CreateSessionArgs_V2
{
    UINT32                      gfeMonitorUsage;           // [in]
    UINT32                      shadowPlayFlags;           // [in]
};

struct CreateSessionArgs_V4: CreateSessionArgs_V3
{
    HANDLE                      hScreenshotCapture;         // [in]
    UINT32                      screenshotVersion;          // [in]
};

struct CreateSessionArgs_V5 : CreateSessionArgs_V4
{
    OSCSurfaceInfo             oscSurfaceInfo;              // [in]
};

struct DestroySessionArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    
};
    
struct GetSessionParamArgs
{
    UINT32                      ver;                  // [in] Set this to DDISHIM_GETSESSIONPARAMARGS_VER
    HANDLE                      hDevice;              // [in] Optional. Can be NULL  
    
};

enum DdiGetSessionParamType
{
    DdiGetSessionParamType_Unknown  = 0,
    DdiGetSessionParamType_InputRedirectionStatus  = 1,     // Input redirection status. 0 - OFF, 1 - ON.
    // <append here>

    DdiGetSessionParamType_Max  = 0,
};

struct GetSessionParamArgs_V2: GetSessionParamArgs
{
    UINT32                      paramId;        // [in] parameter ID from DdiGetSessionParamType enum
    UINT32                      dataSize;       // [in] size of pData
    void                        *pData;         // [out] result data
};
    
struct SetSessionParamArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
};

struct DdiOpenAdapterArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hAdapter;
};

struct DdiCreateDeviceArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    UINT32                      d3dVersion;                  // [in] 
    HANDLE                      hLogoRenderer;  
    UINT32                      rendererVersion;
    UINT32                      isSurround;           // [in] 
    UINT32                      isInBlackList;        // [in] 
    UINT32                      isOptimus;        // [in] 
    wchar_t                     *pszProfileName;
};
    

struct DdiDestroyDeviceArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
};
    
struct DdiCreateResourceArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    HANDLE                      hResource;    
    UINT32                      isPrimary;           // [in] 
    UINT32                      format;           // [in] 
    UINT32                      width;           // [in] 
    UINT32                      height;           // [in] 
    UINT32                      refreshRate;           // [in] 
    UINT32                      vidPnSourceId;
    UINT32                      flags1;
    UINT32                      flags2;
    UINT32                      rotation;
    UINT32                      stereo;
};
    
struct DdiDestroyResourceArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    HANDLE                      hResource;    
    
};
    
struct DdiSetDisplayModeArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;    
    HANDLE                      hResource;    
};

   
struct DdiPrePresentArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    HANDLE                      hResource;    
    UINT32                      flip;                 // [in] 
    UINT32                      isFullScreen;         // [in] 
    UINT32                      width;                // [in] 
    UINT32                      height;               // [in] 
    UINT32                      stereo;               // [in]
};

struct DdiPrePresentArgs_V2 : DdiPrePresentArgs
{
    UINT32                      setFullScreenFlag;    // [in]
};

struct DdiPrePresentArgs_V3 : DdiPrePresentArgs_V2
{
    DirectFlipState             lwrrentFlipState;     // [in]
};

struct DdiPrePresentArgs_V4 : DdiPrePresentArgs_V3
{
    UINT32                      SrcSubResourceIndex;     // [in]
};

struct DdiPrePresentArgs_V5 : DdiPrePresentArgs_V4
{
    UINT32                      rendererVersion;
    HANDLE                      hRenderer;
};

struct DdiPostPresentArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    UINT32                      flip;                 // [in] 
};

struct DdiPostPresentArgs_V2 : DdiPostPresentArgs
{
    UINT32                      SaveScreenshot;    // [in]
};

struct DdiBltArgs
{
    UINT32                      ver;                  // [in] 
    HANDLE                      hDevice;
    HANDLE                      hSrcResource;  
    HANDLE                      hDstResource;  
    RECT                        srcRect;
    RECT                        dstRect;
    UINT32                      flags;
};

static void * const SP_DUMMY_HANDLE = UintToPtr(0xFFFFFFFF);

struct ScreeshotCaptureArgs
{
    UINT32     version;
    HANDLE     hResource;
    HANDLE     hDevice;
    UINT32     SrcSubResourceIndex;
    UINT32     errorcode;
    UINT32     width;
    UINT32     height;
    UINT32     pitch;
    UINT32     bpp;   
    UINT32     format;
    BYTE       *pImageData;
};

struct CreateCommandQueueArgs
{
    UINT32 version;
    HANDLE hDevice;
    HANDLE hLogoRenderer;
    UINT32 rendererVersion;
};

struct DestroyCommandQueueArgs
{
    UINT32 version;
    HANDLE hDevice;
    HANDLE hLogoRenderer;
    UINT32 rendererVersion;
};

#define SP_SCREENSHOTCAPTUREARGS_VER         SP_MAKE_VERSION(ScreeshotCaptureArgs, 1)
#define SP_CREATECOMMANDQUEUEARGS_VER       SP_MAKE_VERSION(CreateCommandQueueArgs, 1)
#define SP_DESTROYCOMMANDQUEUEARGS_VER       SP_MAKE_VERSION(DestroyCommandQueueArgs, 1)

class CScreenshotCaptureDelegate
{

public:
    CScreenshotCaptureDelegate() { };
    virtual ~CScreenshotCaptureDelegate(){};
    virtual HRESULT CaptureFrame(ScreeshotCaptureArgs *pCaptureArgs)=0;
};

#define SP_SCREENSHOTCAPTURE_VER         SP_MAKE_VERSION(CScreenshotCaptureDelegate, 1)


// IShadowPlayDdiShim1
struct IShadowPlayDdiShimVer1
{
    typedef HRESULT (__stdcall * CheckStatus_FuncType)(CheckStatusArgs *pArgs);
    typedef HRESULT (__stdcall * CreateSession_FuncType)(HANDLE *pHandle, CreateSessionArgs *pArgs);
    typedef HRESULT (__stdcall * DestroySession_FuncType)(HANDLE handle,  DestroySessionArgs *pArgs);
    typedef HRESULT (__stdcall * GetSessionParam_FuncType)(HANDLE handle,  GetSessionParamArgs_V2 *pArgs);
    typedef HRESULT (__stdcall * SetSessionParam_FuncType)(HANDLE handle,  SetSessionParamArgs *pArgs);
    typedef HRESULT (__stdcall * DdiOpenAdapter_FuncType)(HANDLE handle,  DdiOpenAdapterArgs *pArgs);
    typedef HRESULT (__stdcall * DdiCreateDevice_FuncType)(HANDLE handle,  DdiCreateDeviceArgs *pArgs);
    typedef HRESULT (__stdcall * DdiDestroyDevice_FuncType)(HANDLE handle,  DdiDestroyDeviceArgs *pArgs);
    typedef HRESULT (__stdcall * DdiCreateResource_FuncType)(HANDLE handle,  DdiCreateResourceArgs *pArgs);
    typedef HRESULT (__stdcall * DdiDestroyResource_FuncType)(HANDLE handle,  DdiDestroyResourceArgs *pArgs);
    typedef HRESULT (__stdcall * DdiPreSetDisplayMode_FuncType)(HANDLE handle,  DdiSetDisplayModeArgs *pArgs);
    typedef HRESULT (__stdcall * DdiPostSetDisplayMode_FuncType)(HANDLE handle,  DdiSetDisplayModeArgs *pArgs);
    typedef HRESULT (__stdcall * DdiPrePresent_FuncType)(HANDLE handle,  DdiPrePresentArgs *pArgs);
    // After LwFBCBlt ?
    typedef HRESULT (__stdcall * DdiPostPresent_FuncType)(HANDLE handle,  DdiPostPresentArgs *pArgs);
    typedef HRESULT (__stdcall * DdiBlt_FuncType) (HANDLE handle,  DdiBltArgs *pArgs);

    // Runs Capture Server
    CheckStatus_FuncType            CheckStatus;
    CreateSession_FuncType          CreateSession;  
    DestroySession_FuncType         DestroySession;     
    GetSessionParam_FuncType        GetSessionParam;    
    SetSessionParam_FuncType        SetSessionParam;

    DdiOpenAdapter_FuncType         DdiOpenAdapter;   
    DdiCreateDevice_FuncType        DdiCreateDevice;   
    DdiDestroyDevice_FuncType       DdiDestroyDevice;   
    DdiCreateResource_FuncType      DdiCreateResource;  
    DdiDestroyResource_FuncType     DdiDestroyResource; 
    DdiPreSetDisplayMode_FuncType   DdiPreSetDisplayMode;  
    DdiPostSetDisplayMode_FuncType  DdiPostSetDisplayMode;  
    DdiPrePresent_FuncType          DdiPrePresent;  
    DdiPostPresent_FuncType         DdiPostPresent;  
    DdiBlt_FuncType                 DdiBlt;         

};

// IShadowPlayDdiShim2
struct IShadowPlayDdiShimVer2 : IShadowPlayDdiShimVer1
{
    typedef HRESULT (__stdcall * CreateSession_V2_FuncType)(HANDLE *pHandle, CreateSessionArgs_V3 *pArgs);
    typedef HRESULT (__stdcall * DdiPrePresent_V2_FuncType)(HANDLE handle,  DdiPrePresentArgs_V3 *pArgs);

    CreateSession_V2_FuncType       CreateSession_V2;

    DdiPrePresent_V2_FuncType       DdiPrePresent_V2;
};

// IShadowPlayDdiShim3
struct IShadowPlayDdiShimVer3 : IShadowPlayDdiShimVer2
{
    typedef HRESULT (__stdcall * ReleaseInterface_FuncType)(HANDLE handle);

    ReleaseInterface_FuncType       ReleaseInterface;
};

// IShadowPlayDdiShim4
struct IShadowPlayDdiShimVer4 : IShadowPlayDdiShimVer3
{
    typedef HRESULT(__stdcall * DdiCreateCommandQueue_FuncType)(HANDLE handle, CreateCommandQueueArgs *pArgs);
    typedef HRESULT(__stdcall * DdiDestroyCommandQueue_FuncType)(HANDLE handle, DestroyCommandQueueArgs *pArgs);

    DdiCreateCommandQueue_FuncType       DdiCreateCommandQueue;
    DdiDestroyCommandQueue_FuncType       DdiDestroyCommandQueue;
};

#define DDISHIM_CHECKSTATUSARGS_VER         SP_MAKE_VERSION(CheckStatusArgs, 1)
#define DDISHIM_CREATESESSIONARGS_VER       SP_MAKE_VERSION(CreateSessionArgs, 1)
#define DDISHIM_DESTROYSESSIONARGS_VER      SP_MAKE_VERSION(DestroySessionArgs, 1)
#define DDISHIM_GETSESSIONPARAMARGS_VER     SP_MAKE_VERSION(GetSessionParamArgs, 1)
#define DDISHIM_SETSESSIONPARAMARGS_VER     SP_MAKE_VERSION(SetSessionParamArgs, 1)

#define DDISHIM_DDIOPENADAPTERARGS_VER      SP_MAKE_VERSION(DdiOpenAdapterArgs, 1)
#define DDISHIM_DDICREATEDEVICEARGS_VER     SP_MAKE_VERSION(DdiCreateDeviceArgs, 1)
#define DDISHIM_DDIDESTROYDEVICEARGS_VER    SP_MAKE_VERSION(DdiDestroyDeviceArgs, 1)
#define DDISHIM_DDICREATERESOURCEARGS_VER   SP_MAKE_VERSION(DdiCreateResourceArgs, 1)
#define DDISHIM_DDIDESTROYRESOURCEARGS_VER  SP_MAKE_VERSION(DdiDestroyResourceArgs, 1)
#define DDISHIM_DDISETDISPLAYMODEARGS_VER   SP_MAKE_VERSION(DdiSetDisplayModeArgs, 1)
#define DDISHIM_DDIPREPRESENTARGS_VER       SP_MAKE_VERSION(DdiPrePresentArgs, 1)
#define DDISHIM_DDIPOSTPRESENTARGS_VER      SP_MAKE_VERSION(DdiPostPresentArgs, 1)
#define DDISHIM_DDIBLTARGS_VER              SP_MAKE_VERSION(DdiBltArgs, 1)

#define DDISHIM_DDIPOSTPRESENTARGS_VER_2    SP_MAKE_VERSION(DdiPostPresentArgs_V2, 1)
#define DDISHIM_CREATESESSIONARGS_VER_2     SP_MAKE_VERSION(CreateSessionArgs_V2, 1)

#define DDISHIM_CREATESESSIONARGS_VER_3     SP_MAKE_VERSION(CreateSessionArgs_V3, 1)
#define DDISHIM_CREATESESSIONARGS_VER_4     SP_MAKE_VERSION(CreateSessionArgs_V4, 1)
#define DDISHIM_CREATESESSIONARGS_VER_5     SP_MAKE_VERSION(CreateSessionArgs_V5, 1)
#define DDISHIM_CREATESESSIONARGS_VER_6     SP_MAKE_VERSION(CreateSessionArgs_V5, 6) //To add support for explicit SLI feature

#define DDISHIM_DDIPREPRESENTARGS_VER_2     SP_MAKE_VERSION(DdiPrePresentArgs_V2, 1)
#define DDISHIM_DDIPREPRESENTARGS_VER_3     SP_MAKE_VERSION(DdiPrePresentArgs_V3, 1)
#define DDISHIM_DDIPREPRESENTARGS_VER_4     SP_MAKE_VERSION(DdiPrePresentArgs_V4, 1)
#define DDISHIM_DDIPREPRESENTARGS_VER_5     SP_MAKE_VERSION(DdiPrePresentArgs_V5, 1)

#define DDISHIM_GETSESSIONPARAMARGS_VER_2     SP_MAKE_VERSION(GetSessionParamArgs_V2, 2) 

#define SHADOWPLAY_DDISHIM_INTERFACE_VER_1      SP_MAKE_VERSION(IShadowPlayDdiShimVer1, 1)
#define SHADOWPLAY_DDISHIM_INTERFACE_VER_2      SP_MAKE_VERSION(IShadowPlayDdiShimVer2, 1)
#define SHADOWPLAY_DDISHIM_INTERFACE_VER_3      SP_MAKE_VERSION(IShadowPlayDdiShimVer3, 1)
#define SHADOWPLAY_DDISHIM_INTERFACE_VER_4      SP_MAKE_VERSION(IShadowPlayDdiShimVer3, 2) // To accomodate CreateSessionArgs_V3
#define SHADOWPLAY_DDISHIM_INTERFACE_VER_5      SP_MAKE_VERSION(IShadowPlayDdiShimVer3, 3) /* To move game profile name justification code (removing
                                                                                              special characters) from driver side to GFE side*/
#define SHADOWPLAY_DDISHIM_INTERFACE_VER_6      SP_MAKE_VERSION(IShadowPlayDdiShimVer4, 4) // DdiCreateCommandQueue API

#define SHADOWPLAY_DDISHIM_INTERFACE_VER        SHADOWPLAY_DDISHIM_INTERFACE_VER_4

HRESULT __stdcall QueryShadowPlayDdiShimStatus(UINT32 *pUseDdiShim, UINT32 *pIsShadowPlayAllowed);

HRESULT __stdcall QueryShadowPlayDdiShimInterface(int version, void **pInterface);

typedef HRESULT (__stdcall * QueryShadowPlayDdiShimStatus_FuncType)(UINT32 *pUseDdiShim, UINT32 *pIsShadowPlayAllowed);

typedef HRESULT (__stdcall * QueryShadowPlayDdiShimInterface_FuncType)(int version, void **pInterface);


