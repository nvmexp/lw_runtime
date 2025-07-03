#pragma once

#include <string>
#include <d3d11.h>
#include "darkroom/PixelFormat.h"

struct AnselSDKCaptureParameters;

#define APP_PATH_MAXLEN 1024 // This should ONLY be used with the GetModuleFileName() string input. More info here: https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file#maximum-path-length-limitation

// Storage classes
class AnselResourceData
{
public:
    ID3D11Texture2D* pTexture2D;
    DWORD width;
    DWORD height;
    DWORD sampleCount;
    DWORD sampleQuality;
    DXGI_FORMAT format;
    ID3D11ShaderResourceView * pSRV;
    ID3D11RenderTargetView * pRTV;
    ID3D11DepthStencilView * pDSV;
};

class AnselSharedResourceData : public AnselResourceData
{
public:
    IDXGIKeyedMutex* pTexture2DMutex;
    HANDLE sharedHandle;
    ID3D11UnorderedAccessView * pUAV;
};

struct AnselEffectState
{
    ID3D11VertexShader * pVertexShader;
    ID3D11PixelShader * pPixelShader;
    ID3D11RasterizerState * pRasterizerState;
    ID3D11DepthStencilState * pDepthStencilState;
    ID3D11SamplerState * pSamplerState;
    ID3D11BlendState * pBlendState;
};

struct AnselResource
{
    // TODO avoroshilov: unify (Data, Key, bool) to one structure
    AnselSharedResourceData toServerRes;
    AnselSharedResourceData toClientRes;
    DWORD toServerWaitKey, toClientWaitKey;
    bool toServerAcquired, toClientAcquired;
};

struct CREATESHAREDRESOURCEDATA
{
    AnselSharedResourceData * pSharedResourceData;
    DWORD overrideFormat;
    DWORD overrideBindFlags;
    DWORD overrideSampleCount;
};

class ShotSaver
{
public:
    virtual HRESULT saveShot(const AnselSDKCaptureParameters &captureParams, bool forceRGBA8Colwersion, const std::wstring& shotName, bool useAdditionalResource = false) = 0;
};

enum struct ShotType
{
    kNone = 0,
    kRegular,
    kRegularUI,
    kStereo,
    kHighRes,
    k360,
    k360Stereo,

    kNumEntries
};

struct UISpecificTelemetryData
{
    ShotType kindOfShot = ShotType::kNone;
    unsigned long highresMult = 0;
    unsigned long long resolution360 = 0;
    float fov = 0.0f;
    float roll = 0.0f;
    bool isShotHDR = false;
    bool usedGamepadForUIDuringTheSession = false;
};

enum class FatalErrorCode
{
    kGeneric = 1,
    kInitFail = 2,
    kReadbackCreateFail = 3,
    kModsDisabledFail = 4,
    kModsSettingFail = 5,
    kHashFail = 6,
    kPostProcessFail = 7,
    kFrameProcessFail = 8,
    kOtherException = 9,

    kNUM_ENTRIES
};
