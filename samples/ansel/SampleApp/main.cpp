#include <iostream>
#include <d3d11.h>
#include <lwapi.h>
#include <LwApiDriverSettings.h>
#include <sstream>

#include <d3dcompiler.h>
#include <windows.h>
#include <lodepng.h>

// This app demonstrates how to enable and configure Ansel from LwAPI - See
// ApplyDRS, RestoreDRS, SaveDRS, and ConfigureAnsel in the SampleApp class.
// This app renders a texture quad on screen and requires a png file named
// "image.png" to be in the app's working directory.

#define CONFIG_FULLSCREEN 0
#define CONFIG_WINDOW_WIDTH 800
#define CONFIG_WINDOW_HEIGHT 600

#ifdef _DEBUG
#define HandleFailure() __debugbreak(); return status;
#else
#define HandleFailure() return status;
#endif

#define SAFE_RELEASE(x) if (x) x->Release();

HWND g_hWnd = 0;
HDC g_hDC = 0;
MSG msg;

struct RenderEffectState
{
    ID3D11VertexShader * pVertexShader;
    ID3D11PixelShader * pPixelShader;
    ID3D11RasterizerState * pRasterizerState;
    ID3D11DepthStencilState * pDepthStencilState;
    ID3D11SamplerState * pSamplerState;
    ID3D11BlendState * pBlendState;
};

class SampleApp
{
    HWND m_hWnd;
    
    unsigned int m_windowWidth;
    unsigned int m_windowHeight;
public:

    IDXGIFactory * m_pFactory;
    IDXGIAdapter * m_pAdapter;
    IDXGIOutput * m_pOutput;
    IDXGISwapChain * m_pSwapChain;
    ID3D11Device * m_pDevice;
    ID3D11DeviceContext * m_pContext;
    ID3D11Texture2D * m_pTexRT;
    ID3D11RenderTargetView * m_pRTV;
    ID3D11Texture2D * m_pTexDepth;
    ID3D11DepthStencilView * m_pDSV;
    ID3D11Texture2D * m_pImageTexture2D;
    ID3D11ShaderResourceView * m_pImageSRV;

    RenderEffectState m_renderEffectState;
    SampleApp();
    HRESULT CreatePassthroughEffect(RenderEffectState * pOut);
    HRESULT Init(HWND hWnd, unsigned int windowWidth, unsigned int windowHeight);
    HRESULT Frame();
    void Destroy();
    
    LwU32 m_savedAnselEnable;
    bool m_bFoundAnselEnable;

    void ApplyDRS();
    void RestoreDRS();
    void SaveDRS();
    void ConfigureAnsel();
    void ReportLwAPIError(LwAPI_Status status);
};

bool ProcessMessage(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

SampleApp::SampleApp()
:
m_pFactory(0),
m_pAdapter(0),
m_pOutput(0),
m_pSwapChain(0),
m_pDevice(0),
m_pContext(0),
m_pTexRT(0),
m_pRTV(0),
m_pTexDepth(0),
m_pDSV(0),
m_pImageTexture2D(0),
m_pImageSRV(0),
m_hWnd(0),
m_windowWidth(0),
m_windowHeight(0),
m_savedAnselEnable(0),
m_bFoundAnselEnable(false)
{}

HRESULT SampleApp::CreatePassthroughEffect(RenderEffectState * pOut)
{
    D3D11_RASTERIZER_DESC rastStateDesc = 
    {
        D3D11_FILL_SOLID,          //FillMode;
        D3D11_LWLL_BACK,           //LwllMode;
        FALSE,                     //FrontCounterClockwise;
        0,                         //DepthBias;
        0.0f,                      //DepthBiasClamp;
        0.0f,                      //SlopeScaledDepthBias;
        TRUE,                      //DepthClipEnable;
        FALSE,                     //ScissorEnable;
        FALSE,                     //MultisampleEnable;
        FALSE                      //AntialiasedLineEnable;
    };

    ID3D11RasterizerState * pRasterizerState;
    HRESULT status = S_OK;
    if (!SUCCEEDED(status = m_pDevice->CreateRasterizerState(&rastStateDesc, &pRasterizerState)))
    {
        HandleFailure();
    }

    D3D11_DEPTH_STENCIL_DESC dsStateDesc;
    memset(&dsStateDesc, 0, sizeof(dsStateDesc));
    dsStateDesc.DepthEnable = FALSE;
    dsStateDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsStateDesc.DepthFunc = D3D11_COMPARISON_ALWAYS;
    dsStateDesc.StencilEnable = FALSE;
    dsStateDesc.StencilReadMask = 0xFF;
    dsStateDesc.StencilWriteMask = 0xFF;
    dsStateDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    dsStateDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

    ID3D11DepthStencilState * pDepthStencilState;
    if (!SUCCEEDED(status = m_pDevice->CreateDepthStencilState(&dsStateDesc, &pDepthStencilState)))
    {
        HandleFailure();
    }

    // Vertex Shader
    ID3D11VertexShader          *pVS = NULL;
    ID3D10Blob                  *pVSBlob = NULL;
    ID3D10Blob                  *pVSBlobErrors = NULL;

    const char vsText[] =
        "struct Output                                                                                  \n"
        "{                                                                                              \n"
        "   float4 position_cs : SV_POSITION;                                                           \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "Output Main(uint id: SV_VertexID)                                                              \n"
        "{                                                                                              \n"
        "   Output output;                                                                              \n"
        "                                                                                               \n"
        "   output.texcoord = float2((id << 1) & 2, id & 2);                                            \n"
        "   output.position_cs = float4(output.texcoord * float2(2, -2) + float2(-1, 1), 0, 1);         \n"
        "                                                                                               \n"
        "   return output;                                                                              \n"
        "}                                                                                              \n";
    
    if (!SUCCEEDED(status = D3DCompile(vsText, sizeof(vsText)-1, NULL, NULL, NULL, "Main", "vs_4_0", 0, 0, &pVSBlob, NULL)))
    {
        char * error = (char *) pVSBlobErrors->GetBufferPointer();
        HandleFailure();
    }

    if (!SUCCEEDED(status = m_pDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &pVS)))
    {
        HandleFailure();
    }

    if (pVSBlob) pVSBlob->Release();
    if (pVSBlobErrors) pVSBlobErrors->Release();

    // Pixel Shader
    ID3D11PixelShader   *pPS = NULL;
    ID3D10Blob          *pPSBlob = NULL;
    ID3D10Blob          *pPSBlobErrors = NULL;
    const char psText[] =
        "struct VSOut                                                                     \n"
        "{                                                                                \n"
        "    float4 position : SV_Position;                                               \n"
        "    float2 texcoord: TexCoord;                                                   \n"
        "};                                                                               \n"
        "                                                                                 \n"
        "Texture2D txDiffuse : register( t0 );                                            \n"
        "SamplerState samLinear : register( s0 );                                         \n"
        "                                                                                 \n"
        "float4 PS( VSOut frag ): SV_Target                                               \n"
        "{                                                                                \n"
        "    float4 clr = txDiffuse.Sample(samLinear, frag.texcoord);                     \n"
        "                                                                                 \n"
        "    return clr;                                                                  \n"
        "}                                                                                \n";
    
    if (!SUCCEEDED(status = D3DCompile(psText, sizeof(psText)-1, NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pPSBlobErrors)))
    {
        char * error = (char *) pPSBlobErrors->GetBufferPointer();
        HandleFailure();
    }
    if (!SUCCEEDED(status = m_pDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &pPS)))
    {
        HandleFailure();
    }
    
    if (pPSBlob) pPSBlob->Release();
    if (pPSBlobErrors) pPSBlobErrors->Release();

    ID3D11SamplerState * pSamplerState = NULL;
    D3D11_SAMPLER_DESC samplerState;
    memset(&samplerState, 0, sizeof(samplerState));
    samplerState.Filter = D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR;
    samplerState.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerState.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerState.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerState.MipLODBias = 0;
    samplerState.MaxAnisotropy = 1;
    samplerState.ComparisonFunc = D3D11_COMPARISON_EQUAL;
    samplerState.BorderColor[0] = 0.0f;
    samplerState.BorderColor[1] = 0.0f;
    samplerState.BorderColor[2] = 0.0f;
    samplerState.BorderColor[3] = 0.0f;
    samplerState.MinLOD = 0;
    samplerState.MaxLOD = 0;
    
    if (!SUCCEEDED(status = m_pDevice->CreateSamplerState(&samplerState, &pSamplerState)))
    {
        HandleFailure();
    }
    
    pOut->pVertexShader = pVS;
    pOut->pPixelShader = pPS;
    pOut->pRasterizerState = pRasterizerState;
    pOut->pDepthStencilState = pDepthStencilState;
    pOut->pSamplerState = pSamplerState;

    pOut->pBlendState = NULL;
    return S_OK;
}

void SampleApp::SaveDRS()
{
    LwAPI_Initialize();
    LwAPI_Status ret;
    LwDRSSessionHandle hSession;
    LwDRSProfileHandle hProfile;

    LwU32 anselEnableID = ANSEL_ENABLE_ID;
    
    ret = LwAPI_DRS_CreateSession(&hSession);
    ret = LwAPI_DRS_LoadSettings(hSession);
    ret = LwAPI_DRS_GetBaseProfile(hSession, &hProfile); 
    if (ret != LWAPI_OK) ReportLwAPIError(ret);

    LWDRS_SETTING lwApiSetting = {0};
    {
        //ANSELENABLE
        lwApiSetting.version = LWDRS_SETTING_VER;
        lwApiSetting.settingId = anselEnableID;
        lwApiSetting.settingType = LWDRS_DWORD_TYPE;

        ret = LwAPI_DRS_GetSetting(hSession, hProfile, anselEnableID, &lwApiSetting);
        if (ret != LWAPI_OK && ret != LWAPI_SETTING_NOT_FOUND) ReportLwAPIError(ret);

        m_bFoundAnselEnable = ret != LWAPI_SETTING_NOT_FOUND;
        if (m_bFoundAnselEnable)
        {
            m_savedAnselEnable = lwApiSetting.u32LwrrentValue;
        }
    }
    
    ret = LwAPI_DRS_DestroySession(hSession);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);
}

void SampleApp::RestoreDRS()
{
    LwAPI_Initialize();
    LwAPI_Status ret;
    LwDRSSessionHandle hSession;
    LwDRSProfileHandle hProfile;

    ret = LwAPI_DRS_CreateSession(&hSession);
    ret = LwAPI_DRS_LoadSettings(hSession);
    ret = LwAPI_DRS_GetBaseProfile(hSession, &hProfile); 
    if (ret != LWAPI_OK) ReportLwAPIError(ret);

    LwU32 anselEnableID = ANSEL_ENABLE_ID;

    ret = LwAPI_DRS_DeleteProfileSetting(hSession, hProfile, anselEnableID);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);

    ret = LwAPI_DRS_SaveSettings(hSession);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);
    ret = LwAPI_DRS_DestroySession(hSession);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);
}

void SampleApp::ApplyDRS()
{
    LwAPI_Initialize();
    LwAPI_Status ret;
    LwDRSSessionHandle hSession;
    LwDRSProfileHandle hProfile;

    ret = LwAPI_DRS_CreateSession(&hSession);
    ret = LwAPI_DRS_LoadSettings(hSession);
    ret = LwAPI_DRS_GetBaseProfile(hSession, &hProfile); 
    if (ret != LWAPI_OK) ReportLwAPIError(ret);

    LwU32 anselEnableID = ANSEL_ENABLE_ID;
    LwU32 anselEnableIDValue = 1;

    {
        //ANSELENABLE
        LWDRS_SETTING lwApiSetting = {0};
        lwApiSetting.version = LWDRS_SETTING_VER;
        lwApiSetting.settingId = anselEnableID;
        lwApiSetting.settingType = LWDRS_DWORD_TYPE;
        lwApiSetting.u32LwrrentValue = anselEnableIDValue;

        ret = LwAPI_DRS_SetSetting(hSession, hProfile, &lwApiSetting);
        if (ret != LWAPI_OK) ReportLwAPIError(ret);
        ret = LwAPI_DRS_SaveSettings(hSession);
    }
    
    ret = LwAPI_DRS_SaveSettings(hSession);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);
    ret = LwAPI_DRS_DestroySession(hSession);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);
}

void SampleApp::ConfigureAnsel()
{
    LWAPI_ANSEL_CONFIGURATION_STRUCT anselConfigStruct;
    memset(&anselConfigStruct, 0, sizeof(anselConfigStruct));
    anselConfigStruct.version = LWAPI_ANSEL_CONFIGURATION_STRUCT_VER1;
    anselConfigStruct.hotkeyModifier = LWAPI_ANSEL_HOTKEY_MODIFIER_CTRL;
    anselConfigStruct.keyEnable = VK_F4;

    // TODO: Get FeatureIds from somewhere
    LWAPI_ANSEL_FEATURE_CONFIGURATION_STRUCT anselFeatureConfigStruct[4];
    anselFeatureConfigStruct[0].featureId = LWAPI_ANSEL_FEATURE_BLACK_AND_WHITE;
    anselFeatureConfigStruct[0].featureState = LWAPI_ANSEL_FEATURE_STATE_ENABLE;
    anselFeatureConfigStruct[0].hotkey = VK_F5;
    
    anselFeatureConfigStruct[1].featureId = LWAPI_ANSEL_FEATURE_HUDLESS;
    anselFeatureConfigStruct[1].featureState = LWAPI_ANSEL_FEATURE_STATE_DISABLE;
    anselFeatureConfigStruct[1].hotkey = VK_F6;
    
    anselFeatureConfigStruct[2].featureId = (LWAPI_ANSEL_FEATURE) 3;
    anselFeatureConfigStruct[2].featureState = LWAPI_ANSEL_FEATURE_STATE_DISABLE;
    anselFeatureConfigStruct[2].hotkey = VK_F7;
    
    anselFeatureConfigStruct[3].featureId = (LWAPI_ANSEL_FEATURE) 4;
    anselFeatureConfigStruct[3].featureState = LWAPI_ANSEL_FEATURE_STATE_DISABLE;
    anselFeatureConfigStruct[3].hotkey = VK_F8;

    anselConfigStruct.numAnselFeatures = 4;
    anselConfigStruct.pAnselFeatures = anselFeatureConfigStruct;
    
    LwAPI_Status ret = LWAPI_OK;
    ret = LwAPI_D3D_ConfigureAnsel(m_pDevice, &anselConfigStruct);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);
}

void SampleApp::ReportLwAPIError(LwAPI_Status status)
{
   LwAPI_ShortString szDesc = {0};
   LwAPI_GetErrorMessage(status, szDesc);
   MessageBox(m_hWnd, (LPCWSTR) szDesc, L"Error", MB_OK);
}

HRESULT LoadFile(const char * path, unsigned int * pWidth, unsigned int * pHeight, unsigned char ** pImageData)
{
    //Read from file.
    unsigned char * fileBuffer = 0;
    unsigned int bufferSize = 0;
    {
        FILE* file;
        long fileSize;

        file = fopen(path, "rb");
        if (!file) return NULL;

        /*get filesize:*/
        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        rewind(file);

        /*read contents of the file into the vector*/
        if (fileSize)
        {
            fileBuffer = new unsigned char[(size_t)fileSize];
        }
        if (fileSize && fileBuffer)
        {
            bufferSize = (unsigned int)fread(fileBuffer, 1, (size_t)fileSize, file);
        }

        fclose(file);
    }

    //Decode PNG data.
    if (fileBuffer)
    {
        unsigned error;
        unsigned char* imageData;
        unsigned int width, height;
        error = LodePNG_decode(&imageData, &width, &height, fileBuffer, bufferSize, 6, 8);

        delete[] fileBuffer;

        if (!error && imageData)
        {
            *pWidth = width;
            *pHeight = height;
            *pImageData = imageData;
        }
    }
    return S_OK;
}

HRESULT SampleApp::Init(HWND hWnd, unsigned int windowWidth, unsigned int windowHeight)
{
    m_hWnd = hWnd;
    m_windowWidth = windowWidth;
    m_windowHeight = windowHeight;
    
    SaveDRS();
    ApplyDRS();

    unsigned int imageWidth = 0;
    unsigned int imageHeight = 0;
    unsigned char * pImageData = NULL;
    
    HRESULT status = S_OK;
    if (!SUCCEEDED(status = LoadFile("image.png", &imageWidth, &imageHeight, &pImageData)))
    {
        HandleFailure();
    }
    
    if (!SUCCEEDED(status = CreateDXGIFactory(__uuidof(IDXGIFactory), (void **) &m_pFactory)))
    {
        HandleFailure();
    }

    if (!SUCCEEDED(status = m_pFactory->EnumAdapters(0, &m_pAdapter)))
    {
        HandleFailure();
    }

    if (!SUCCEEDED(status = m_pAdapter->EnumOutputs(0, &m_pOutput)))
    {
        HandleFailure();
    }

	UINT numDisplayModes = 0;
	m_pOutput->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &numDisplayModes, 0);
	
	bool modeFound = false;
	DXGI_MODE_DESC * modes = new DXGI_MODE_DESC[numDisplayModes];
	m_pOutput->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &numDisplayModes, modes);
	for (UINT i = 0; i < numDisplayModes; ++i)
	{
		DXGI_MODE_DESC & current = modes[i];
        std::cout << "Mode: " << current.Width << " x " << current.Height << std::endl;
		if (current.Width == imageWidth && current.Height == imageHeight)
		{
			modeFound = true;
		}
	}
	delete [] modes;

    static const D3D_FEATURE_LEVEL featureLevels [] =
	{
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
		D3D_FEATURE_LEVEL_9_3,
		D3D_FEATURE_LEVEL_9_2,
		D3D_FEATURE_LEVEL_9_1,
	};

	D3D_FEATURE_LEVEL featureLevel;
	UINT numFeatureLevels = sizeof(featureLevels) / sizeof(D3D_FEATURE_LEVEL);
    if (!SUCCEEDED(status = D3D11CreateDevice(m_pAdapter,
                                              D3D_DRIVER_TYPE_UNKNOWN,
                                              NULL,
                                              0,
                                              featureLevels,
                                              numFeatureLevels,
                                              D3D11_SDK_VERSION,
                                              &m_pDevice,
                                              &featureLevel,
                                              &m_pContext)))
    {
        HandleFailure();
    }
    
    static DXGI_SWAP_CHAIN_DESC swapChainDesc =
    {
        {
            imageWidth,
            imageHeight,
            {0, 0}, //AutoDetect
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE,
            DXGI_MODE_SCALING_UNSPECIFIED
        },                                     //BufferDesc
        {
            1,                                 //Count
            0                                  //Quality
        },                                     //SampleDesc
        DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER, //BufferUsage
        2,                                     //BufferCount
        g_hWnd,                                //OutputWindow
        1,                                     //Windowed
        DXGI_SWAP_EFFECT_DISCARD,              //SwapEffect
        DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH //Flags
    };

    if (!SUCCEEDED(status = m_pFactory->CreateSwapChain(m_pDevice, &swapChainDesc, &m_pSwapChain)))
    {
        HandleFailure();
    }

    while (g_hWnd && PeekMessage(&msg, g_hWnd, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

#if CONFIG_FULLSCREEN
	DXGI_MODE_DESC DXGI_MODE_DESC_0 = 
		{imageWidth,
        imageHeight,
        {0, 0}, //AutoDetect
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE,
		DXGI_MODE_SCALING_UNSPECIFIED};
	m_pSwapChain->SetFullscreenState(TRUE, NULL);
	m_pSwapChain->ResizeBuffers(0, imageWidth, imageHeight, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH);
	m_pSwapChain->ResizeTarget(&DXGI_MODE_DESC_0);
	
#else
	m_pSwapChain->SetFullscreenState(FALSE, NULL);
#endif

    while (g_hWnd && PeekMessage(&msg, g_hWnd, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    //Create Render Target Views:

	m_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&m_pTexRT);
	m_pDevice->CreateRenderTargetView(m_pTexRT, NULL, &m_pRTV);
	m_pSwapChain->Release();

	static D3D11_TEXTURE2D_DESC descDepth;
    memset(&descDepth, 0, sizeof(descDepth));
	descDepth.Width = imageWidth;
	descDepth.Height = imageHeight;
	descDepth.MipLevels = 1;
	descDepth.ArraySize = 1;
	descDepth.Format = DXGI_FORMAT_D32_FLOAT;
	descDepth.SampleDesc.Count = 1;
	descDepth.SampleDesc.Quality = 0;
	descDepth.Usage = D3D11_USAGE_DEFAULT;
	descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	descDepth.CPUAccessFlags = 0;
	descDepth.MiscFlags = 0;
	
	m_pDevice->CreateTexture2D(&descDepth, NULL, &m_pTexDepth);

	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
    memset(&descDSV, 0, sizeof(descDSV));
	descDSV.Format = descDepth.Format;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;

	m_pDevice->CreateDepthStencilView(m_pTexDepth, &descDSV, &m_pDSV);
	
	//2. Create the Render Target View To Back Buffer. }
	//3. Set Viewport. {
	
	static D3D11_VIEWPORT viewPortDesc;
	viewPortDesc.Width = imageWidth;
	viewPortDesc.Height = imageHeight;
	viewPortDesc.MinDepth = 0.0f;
	viewPortDesc.MaxDepth = 1.0f;
	viewPortDesc.TopLeftX = 0;
	viewPortDesc.TopLeftY = 0;
	 
	m_pContext->RSSetViewports(1, &viewPortDesc);

    CreatePassthroughEffect(&m_renderEffectState);

    D3D11_TEXTURE2D_DESC textureDesc;
    textureDesc.Width = imageWidth;
    textureDesc.Height = imageHeight;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    textureDesc.CPUAccessFlags = 0;
    textureDesc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA subResData;
    subResData.pSysMem = pImageData;
    subResData.SysMemSlicePitch = NULL;
    subResData.SysMemPitch = imageWidth * 4;
    
    if (!SUCCEEDED(status = m_pDevice->CreateTexture2D(&textureDesc, &subResData, &m_pImageTexture2D)))
    {
        HandleFailure();
    }
    
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    memset(&srvDesc, 0, sizeof(srvDesc));
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels = 1;

    if (!SUCCEEDED(status = m_pDevice->CreateShaderResourceView(m_pImageTexture2D, &srvDesc, &m_pImageSRV)))
    {
        HandleFailure();
    }

    LodePNG_Free(pImageData);
    
    ConfigureAnsel();
    
    RestoreDRS();
    return S_OK;
}

void SampleApp::Destroy()
{
    LwAPI_Unload();

    SAFE_RELEASE(m_renderEffectState.pVertexShader);
    SAFE_RELEASE(m_renderEffectState.pPixelShader);
    SAFE_RELEASE(m_renderEffectState.pRasterizerState);
    SAFE_RELEASE(m_renderEffectState.pDepthStencilState);
    SAFE_RELEASE(m_renderEffectState.pSamplerState);
    SAFE_RELEASE(m_renderEffectState.pBlendState);
    SAFE_RELEASE(m_pFactory);
    SAFE_RELEASE(m_pAdapter);
    SAFE_RELEASE(m_pOutput);
    SAFE_RELEASE(m_pSwapChain);
    SAFE_RELEASE(m_pDevice);
    SAFE_RELEASE(m_pContext);
    SAFE_RELEASE(m_pTexRT);
    SAFE_RELEASE(m_pRTV);
    SAFE_RELEASE(m_pTexDepth);
    SAFE_RELEASE(m_pDSV);
    SAFE_RELEASE(m_pImageTexture2D);
    SAFE_RELEASE(m_pImageSRV);
}
HRESULT SampleApp::Frame()
{
    while (g_hWnd && PeekMessage(&msg, g_hWnd, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    m_pContext->VSSetShader(m_renderEffectState.pVertexShader, NULL, 0);
    m_pContext->PSSetShader(m_renderEffectState.pPixelShader, NULL, 0);
    m_pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_pContext->PSSetShaderResources(0, 1, &m_pImageSRV);
    m_pContext->PSSetSamplers(0, 1, &m_renderEffectState.pSamplerState);
    m_pContext->RSSetState(m_renderEffectState.pRasterizerState);
    m_pContext->OMSetDepthStencilState(m_renderEffectState.pDepthStencilState, 0xFFFFFFFF);
    m_pContext->OMSetRenderTargets(1, &m_pRTV, NULL);
    m_pContext->OMSetBlendState(m_renderEffectState.pBlendState, NULL, 0xffffffff);
    m_pContext->Draw(3, 0);
    m_pSwapChain->Present(1, 0);
    return S_OK;
}

bool ProcessMessage(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
		case WM_KEYDOWN:
		{
			switch(wParam)
			{
				case VK_SPACE:
				{
				}
				break;
				case VK_CONTROL:
				{
				}
				break;
			}
		}
		break;
		case WM_SIZE:
			break;
		default:
			break;
	}
	return false;
}

//d3d9.lib d3d10.lib d3d11.lib dxgi.lib d3dx10.lib
//Mult-Threaded Debug /MTd

bool WindowOpen();
void WindowClose();
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
int main(int argc, char ** argv);

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	if (!ProcessMessage(hWnd, message, wParam, lParam))
	{
		switch (message)
		{
			case WM_KEYDOWN:
			{
				switch(wParam)
				{
					case VK_ESCAPE : WindowClose();
					break;
				}
			}
			break;
			case WM_DESTROY:
			{
				PostQuitMessage(0);
				if (g_hWnd == hWnd)
				{
					g_hWnd = NULL;
				}
			}
			break;
		}
	}
    return DefWindowProc(hWnd, message, wParam, lParam);
}

bool WindowOpen()
{
    HINSTANCE hInstance = GetModuleHandle(NULL);

    WNDCLASSEX wcex;
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = (WNDPROC) WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = 0;
    wcex.hLwrsor = LoadLwrsor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH) (COLOR_WINDOW+1);
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = TEXT("WindowClass");
    wcex.hIconSm = 0;
    RegisterClassEx(&wcex);

    RECT rect = {0, 0, CONFIG_WINDOW_WIDTH, CONFIG_WINDOW_HEIGHT};
    AdjustWindowRect(&rect, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX, FALSE);

#if CONFIG_FULLSCREEN
	g_hWnd = CreateWindow(L"WindowClass",
                          L"Window",
                          WS_EX_TOPMOST | WS_POPUP,
                          CW_USEDEFAULT,
                          CW_USEDEFAULT,
                          CONFIG_WINDOW_WIDTH,
                          CONFIG_WINDOW_HEIGHT,
                          NULL,
                          NULL,
                          hInstance,
                          NULL);
#else
    g_hWnd = CreateWindow(L"WindowClass",
                          L"Window",
                          WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_THICKFRAME | CS_OWNDC,
                          CW_USEDEFAULT,
                          CW_USEDEFAULT,
                          rect.right - rect.left,
                          rect.bottom - rect.top,
                          NULL,
                          NULL,
                          hInstance,
                          NULL);
#endif
    if (!g_hWnd) return false;

    g_hDC = GetDC(g_hWnd);

    ShowWindow(g_hWnd, SW_SHOW);
    UpdateWindow(g_hWnd);

    return true;
}

void WindowClose()
{
    if (g_hWnd)
    {
        DestroyWindow(g_hWnd);
        ReleaseDC(g_hWnd, g_hDC);
	    g_hWnd = 0;
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR pCmdLine, int nCmdShow)
{
    bool success = WindowOpen();
    if (!success)
    {
        return 1;
    }
    SampleApp app;
	if (app.Init(g_hWnd, CONFIG_WINDOW_WIDTH, CONFIG_WINDOW_HEIGHT) == S_OK)
	{

        while (true)
		{
			if (!SUCCEEDED(app.Frame())) break;
			while (g_hWnd && PeekMessage(&msg, g_hWnd, 0, 0, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
			if (!g_hWnd) 
			{
				break;
			}
		}
		WindowClose();
        app.Destroy();
	}
	else
	{
		WindowClose();
		return 1;
	}
	return 0;
}
