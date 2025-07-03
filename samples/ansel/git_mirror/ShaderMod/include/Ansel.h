#ifndef _ANSEL_H_
#define _ANSEL_H_

#ifdef ANSEL_DLL_EXPORTS
#define ANSEL_DLL_API __declspec(dllexport)
#else
#define ANSEL_DLL_API
#endif

#define ANSEL_VERSION_MAJOR 7
#define ANSEL_VERSION_MINOR 1

extern "C"
{
    //*****************************************************************
    // Exported functions
    //*****************************************************************

    typedef void(__cdecl *PFNANSELGETVERSION)(DWORD * pDwMajor, DWORD * pDwMinor);
    ANSEL_DLL_API void __cdecl AnselGetVersion(DWORD * pDwMajor, DWORD * pDwMinor);

    // Called by shim to see if application is allowlisted for Ansel
    typedef bool(__cdecl *PFNANSELENABLECHECK)();
    ANSEL_DLL_API bool __cdecl AnselEnableCheck();
}

#ifndef ANSEL_ALLOWLISTING_ONLY

extern "C"
{
    typedef void * HANSELSERVER;
    typedef void * HCLIENTRESOURCE;
    typedef void * HANSELCLIENT;
    
    // DX12-related
    typedef void * HCMDLIST;
    typedef void * HANSELCMDLIST;
    typedef void * HEXELWTIONCONTEXT;
    struct ANSEL_EXEC_DATA
    {
        HEXELWTIONCONTEXT hExelwtionContext;
        UINT dwCmdQueueId;
        void * pServerData;
    };

    typedef enum
    {
        ANSEL_BLEND_OP_UNKNOWN,
        ANSEL_BLEND_OP_ADD,          //Add source 1 and source 2.
        ANSEL_BLEND_OP_SUBTRACT,     //Subtract source 1 from source 2.
        ANSEL_BLEND_OP_REV_SUBTRACT, //Subtract source 2 from source 1.
        ANSEL_BLEND_OP_MIN,          //Find the minimum of source 1 and source 2.
        ANSEL_BLEND_OP_MAX           //Find the maximum of source 1 and source 2.
    } ANSEL_BLEND_OP;

    typedef enum
    {
        ANSEL_BLEND_UNKNOWN,
        ANSEL_BLEND_ZERO,
        ANSEL_BLEND_ONE,
        ANSEL_BLEND_SRC_COLOR,
        ANSEL_BLEND_ILW_SRC_COLOR,
        ANSEL_BLEND_SRC_ALPHA,
        ANSEL_BLEND_ILW_SRC_ALPHA,
        ANSEL_BLEND_DEST_ALPHA,
        ANSEL_BLEND_ILW_DEST_ALPHA,
        ANSEL_BLEND_DEST_COLOR,
        ANSEL_BLEND_ILW_DEST_COLOR,
        ANSEL_BLEND_SRC_ALPHA_SAT,
        ANSEL_BLEND_BLEND_FACTOR,
        ANSEL_BLEND_ILW_BLEND_FACTOR,
        ANSEL_BLEND_SRC1_COLOR,
        ANSEL_BLEND_ILW_SRC1_COLOR,
        ANSEL_BLEND_SRC1_ALPHA,
        ANSEL_BLEND_ILW_SRC1_ALPHA
    } ANSEL_BLEND;

    typedef enum
    {
        ANSEL_COLOR_WRITE_ENABLE_RED = 1,
        ANSEL_COLOR_WRITE_ENABLE_GREEN = 2,
        ANSEL_COLOR_WRITE_ENABLE_BLUE = 4,
        ANSEL_COLOR_WRITE_ENABLE_ALPHA = 8,
        ANSEL_COLOR_WRITE_ENABLE_ALL = (1|2|4|8),
    } ANSEL_COLOR_WRITE_ENABLE;

    typedef enum
    {
        ANSEL_LWLL_UNKNOWN,
        ANSEL_LWLL_NONE,
        ANSEL_LWLL_FRONT,
        ANSEL_LWLL_BACK
    } ANSEL_LWLL_MODE;

    struct AnselDeviceStates
    {
        BOOL DepthEnable;
        BOOL BlendEnable;
        BOOL StencilEnable;
        ANSEL_LWLL_MODE LwllMode;
        ANSEL_BLEND_OP BlendOp;
        ANSEL_BLEND SrcBlend;
        ANSEL_BLEND SrcBlendAlpha;
        UINT8 RenderTargetWriteMask;
        DWORD numPresentRTs;
        DWORD backbufferWidth;
        DWORD backbufferHeight;
        FLOAT ViewportTopLeftX;
        FLOAT ViewportTopLeftY;
        FLOAT ViewportWidth;
        FLOAT ViewportHeight;
        DWORD RTZeroFormat;
        BOOL  RTZeroIsPresent;

        HCLIENTRESOURCE hLwrrentRTZero;
        HCLIENTRESOURCE hLwrrentDSBuffer;
    };

    //*****************************************************************
    // Exported functions
    //*****************************************************************

    // Called by client to retrieve function table and size
    typedef HRESULT (__cdecl *PFNANSELGETFUNCTIONTABLE)(void * pMem);
    typedef DWORD (__cdecl *PFNANSELGETFUNCTIONTABLESIZE)(void);
    
    ANSEL_DLL_API HRESULT __cdecl AnselGetFunctionTable(void * pMem);
    ANSEL_DLL_API DWORD __cdecl AnselGetFunctionTableSize(void);
    
    //*****************************************************************
    // Callback table functions
    //*****************************************************************

    // Called by client to create the AnselServer
    struct ClientFunctionTable;
    typedef HANSELSERVER (__cdecl *PFNANSELCREATESERVER)(void *, ClientFunctionTable *);
    
    typedef HANSELSERVER(__cdecl *PFNCREATESERVERONADAPTER)(void *, ClientFunctionTable *, LARGE_INTEGER AdapterLuid);
    typedef HRESULT(__cdecl *PFNRESTRICTSERVERCREATION)(void *, bool);

    // Called by driver to release the AnselServer
    typedef void (__cdecl *PFNANSELRELEASESERVER)(HANSELSERVER hAnselServer);
    
    // Takes in a handle to the AnselServer and a handle to the shared resource holding the app's render target data.
    // The AnselServer is expected to execute its post-processing effects with the shared resource as its input.
    typedef HRESULT (__cdecl *PFNANSELEXELWTEPOSTPROCESSING)(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource, DWORD subResIndex);

    // Called by driver to have the AnselServer create a shared resource to hold the app's depth target data
    typedef HRESULT (__cdecl *PFNANSELCREATESHAREDRESOURCE)(HANSELSERVER hAnselServer,
                                                            DWORD width,
                                                            DWORD height,
                                                            DWORD sampleCount,
                                                            DWORD sampleQuality,
                                                            DWORD format,
                                                            HANDLE * pHandle,
                                                            void * pServerPrivateData);

    // Functions called by client to send event notifications to server
    typedef HRESULT (__cdecl *PFNNOTIFYDRAW) (HANSELSERVER hAnselServer);
    typedef HRESULT (__cdecl *PFNNOTIFYDEPTHSTENCILCREATE)(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource);
    typedef HRESULT (__cdecl *PFNNOTIFYDEPTHSTENCILDESTROY)(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource);
    typedef HRESULT (__cdecl *PFNNOTIFYCLIENTRESOURCEDESTROY)(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource);
    typedef HRESULT (__cdecl *PFNNOTIFYRENDERTARGETBIND)(HANSELSERVER hAnselServer, HCLIENTRESOURCE* phClientResource, DWORD dwNumRTs);

    typedef enum
    {
        ANSEL_VIEW_UNKNOWN = 0, // Unknown or previous binding notification
        ANSEL_VIEW_BUFFER = 1,
        ANSEL_VIEW_TEX1D = 2,
        ANSEL_VIEW_TEX1D_ARRAY = 3,
        ANSEL_VIEW_TEX2D = 4,
        ANSEL_VIEW_TEX2D_ARRAY = 5,
        ANSEL_VIEW_TEX2DMS = 6,
        ANSEL_VIEW_TEX2DMS_ARRAY = 7,
        ANSEL_VIEW_TEX3D = 8,
    } ANSEL_VIEW_DIMENSION;

    typedef HRESULT (__cdecl *PFNNOTIFYRENDERTARGETBINDWITHFORMAT)(HANSELSERVER hAnselServer, HCLIENTRESOURCE* phClientResource, DWORD dwNumRTs, const DWORD * pFormats, const ANSEL_VIEW_DIMENSION * pViewDimensions);
    typedef HRESULT (__cdecl *PFNNOTIFYUNORDEREDACCESSBIND)(HANSELSERVER hAnselServer, DWORD startOffset, DWORD count, HCLIENTRESOURCE* phClientResource);
    typedef HRESULT (__cdecl *PFNNOTIFYDEPTHSTENCILBIND)(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource);

    typedef enum
    {
        ANSEL_DEPTH_STENCIL_VIEW_NONE = 0, 
        ANSEL_DEPTH_STENCIL_VIEW_READ_ONLY_DEPTH = 1,
        ANSEL_DEPTH_STENCIL_VIEW_READ_ONLY_STENCIL = 2,
    } ANSEL_DEPTH_STENCIL_VIEW_FLAGS;

    typedef HRESULT (__cdecl *PFNNOTIFYDEPTHSTENCILBINDWITHFORMAT)(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource, DWORD format, ANSEL_VIEW_DIMENSION viewDimension, ANSEL_DEPTH_STENCIL_VIEW_FLAGS viewFlags);
    typedef HRESULT (__cdecl *PFNNOTIFYDEPTHSTENCILCLEAR)(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource);
    typedef HRESULT (__cdecl *PFNNOTIFYRENDERTARGETCLEAR)(HANSELSERVER hAnselServer, HCLIENTRESOURCE hClientResource);
    typedef HRESULT (__cdecl *PFNNOTIFYHOTKEY)(HANSELSERVER hAnselServer, DWORD vkey);

    // DX12 Interfaces
    typedef HRESULT (__cdecl *PFNNOTIFYCMDLISTCREATE12) (HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST *phAnselCmdList);
    typedef HRESULT (__cdecl *PFNNOTIFYCMDLISTDESTROY12) (HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList);
    typedef HRESULT (__cdecl *PFNNOTIFYCMDLISTRESET12) (HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList);
    typedef HRESULT (__cdecl *PFNNOTIFYCMDLISTCLOSE12) (HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList);

    // Notifications for Draw/Present bakes. The implementations of these functions must be reentrant.
    typedef HRESULT (__cdecl *PFNNOTIFYSETRENDERTARGETBAKE12) (HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, void ** ppServerData);
    typedef HRESULT (__cdecl *PFNNOTIFYPRESENTBAKE12) (HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, void ** ppServerData);
    typedef HRESULT (__cdecl *PFNNOTIFYDEPTHSTENCILCLEARBAKE12)(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, HCLIENTRESOURCE hDepthStencil, void ** ppServerData);
    typedef HRESULT (__cdecl *PFNNOTIFYRENDERTARGETCLEARBAKE12)(HANSELSERVER hAnselServer, HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, HCLIENTRESOURCE hDepthStencil, void ** ppServerData);
    typedef HRESULT(__cdecl *PFNNOTIFYSETRENDERTARGETBAKEWITHDEVICESTATES12) (HANSELSERVER hAnselServer, const AnselDeviceStates deviceStates, HANSELCMDLIST hAnselCmdList, void ** ppServerData);

    // Notifications for Draw/Present exelwtion.
    typedef HRESULT (__cdecl *PFNNOTIFYSETRENDERTARGETEXEC12) (HANSELSERVER hAnselServer, ANSEL_EXEC_DATA *pExelwtionContext);
    typedef HRESULT (__cdecl *PFNANSELEXELWTEPOSTPROCESSING12)(HANSELSERVER hAnselServer, ANSEL_EXEC_DATA *pExelwtionContext, HCLIENTRESOURCE hClientResource, DWORD subResIndex);
    typedef HRESULT (__cdecl *PFNNOTIFYDEPTHSTENCILCLEAREXEC12)(HANSELSERVER hAnselServer, ANSEL_EXEC_DATA *pExelwtionContext);
    typedef HRESULT (__cdecl *PFNNOTIFYRENDERTARGETCLEAREXEC12)(HANSELSERVER hAnselServer, ANSEL_EXEC_DATA *pExelwtionContext);
    // End of DX12 Interfaces

    // Called by the driver to update the active GPU mask associated with Ansel's device, only when the client is running in AFR mode of SLI.
    // Required because the driver does not have access to Ansel's device.
    typedef HRESULT(__cdecl *PFNANSELUPDATEGPUMASK)(HANSELSERVER hAnselServer, DWORD activeGPUMask);

    typedef HRESULT (__cdecl *PFNNOTIFYCLIENTMODE)(HANSELSERVER hAnselServer, DWORD clientMode);

    typedef enum _ANSEL_CLIENT_MODE
    {
        ANSEL_CLIENT_MODE_PASSTHROUGH  = 1,  // Clients have passthrough behavior. AnselServer will receive no more notifications.
        ANSEL_CLIENT_MODE_LIGHTWEIGHT  = 2,  // Clients will only report Presents.
    } ANSEL_CLIENT_MODE;
    
    typedef enum _ANSEL_FEATURE
    {
        ANSEL_FEATURE_UNKNOWN         = 0,
        ANSEL_FEATURE_BLACK_AND_WHITE = 1,
        ANSEL_FEATURE_HUDLESS         = 2
    } ANSEL_FEATURE;

    typedef enum _ANSEL_FEATURE_STATE
    {
        ANSEL_FEATURE_STATE_UNKNOWN   = 0,
        ANSEL_FEATURE_STATE_ENABLE    = 1,    //!< Toggle feature on
        ANSEL_FEATURE_STATE_DISABLE   = 2     //!< Toggle feature off
    } ANSEL_FEATURE_STATE;

    typedef enum _ANSEL_HOTKEY_MODIFIER
    {
        ANSEL_HOTKEY_MODIFIER_UNKNOWN = 0,
        ANSEL_HOTKEY_MODIFIER_CTRL    = 1,    //!< Use control in the hotkey combination
        ANSEL_HOTKEY_MODIFIER_SHIFT   = 2,    //!< Use shift in the hotkey combination
        ANSEL_HOTKEY_MODIFIER_ALT     = 3     //!< Use alternate in the hotkey combination
    } ANSEL_HOTKEY_MODIFIER;

    struct AnselFeatureConfig
    {
        ANSEL_FEATURE       featureId;        //Id of feature
        ANSEL_FEATURE_STATE featureState;     //Whether the feature is enabled or not
        UINT                hotkey;           //Optional hotkey associated with this effect
    };

    struct AnselConfig
    {
        UINT version;
        ANSEL_HOTKEY_MODIFIER hotkeyModifier;  //!< Modifier key to use in hotkey combination
        UINT keyEnable;                        //key to enable/disable Ansel
        UINT numAnselFeatures;                 //Number of features in pAnselFeatures
        AnselFeatureConfig * pAnselFeatures;   //Array of feature configurations
    };

    typedef HRESULT (__cdecl *PFNSETCONFIG)(HANSELSERVER hAnselServer, AnselConfig * pConfig);

    // Definition of function table to be returned by PFNANSELGETFUNCTIONTABLE
    struct AnselFunctionTable
    {
        PFNANSELCREATESERVER CreateServer;
        PFNANSELRELEASESERVER ReleaseServer;
        PFNANSELEXELWTEPOSTPROCESSING ExelwtePostProcessing;
        PFNANSELCREATESHAREDRESOURCE CreateSharedResource;
        
        PFNNOTIFYDRAW NotifyDraw;
        PFNNOTIFYDEPTHSTENCILCREATE NotifyDepthStencilCreate;
        PFNNOTIFYDEPTHSTENCILDESTROY NotifyDepthStencilDestroy;
        PFNNOTIFYDEPTHSTENCILBIND NotifyDepthStencilBind;
        PFNNOTIFYRENDERTARGETBIND NotifyRenderTargetBind;
        PFNNOTIFYUNORDEREDACCESSBIND NotifyUnorderedAccessBind;
        PFNNOTIFYCLIENTRESOURCEDESTROY NotifyClientResourceDestroy;
        PFNNOTIFYDEPTHSTENCILCLEAR NotifyDepthStencilClear;

        PFNSETCONFIG SetConfig;
        PFNNOTIFYHOTKEY NotifyHotkey;

        PFNANSELUPDATEGPUMASK UpdateGPUMask;

        PFNNOTIFYCMDLISTCREATE12 NotifyCmdListCreate12;
        PFNNOTIFYCMDLISTDESTROY12 NotifyCmdListDestroy12;
        PFNNOTIFYCMDLISTRESET12 NotifyCmdListReset12;
        PFNNOTIFYCMDLISTCLOSE12 NotifyCmdListClose12;
        PFNNOTIFYSETRENDERTARGETBAKE12 NotifySetRenderTargetBake12;
        PFNNOTIFYSETRENDERTARGETEXEC12 NotifySetRenderTargetExec12;
        PFNNOTIFYPRESENTBAKE12 NotifyPresentBake12;
        PFNANSELEXELWTEPOSTPROCESSING12 ExelwtePostProcessing12;
        PFNNOTIFYDEPTHSTENCILCLEARBAKE12 NotifyDepthStencilClearBake12;
        PFNNOTIFYDEPTHSTENCILCLEAREXEC12 NotifyDepthStencilClearExec12;
         
        PFNNOTIFYCLIENTMODE NotifyClientMode;

        // Only use if Ansel server reports version 5.1 or later
        PFNNOTIFYDEPTHSTENCILBINDWITHFORMAT NotifyDepthStencilBindWithFormat;
        PFNNOTIFYRENDERTARGETBINDWITHFORMAT NotifyRenderTargetBindWithFormat;
        PFNNOTIFYRENDERTARGETCLEAR NotifyRenderTargetClear;
        PFNNOTIFYDEPTHSTENCILCLEARBAKE12 NotifyRenderTargetClearBake12;
        PFNNOTIFYDEPTHSTENCILCLEAREXEC12 NotifyRenderTargetClearExec12;

        PFNCREATESERVERONADAPTER CreateServerOnAdapter;
        PFNRESTRICTSERVERCREATION RestrictServerCreation;

        // Only use if Ansel server reports version 7.0 or later
        PFNNOTIFYSETRENDERTARGETBAKEWITHDEVICESTATES12 NotifySetRenderTargetBakeWithDeviceStates12;
    };

    typedef enum
    {
        ANSEL_TRANSFER_OP_COPY,
        ANSEL_TRANSFER_OP_RESOLVE,
        ANSEL_TRANSFER_OP_MAX
    } ANSEL_TRANSFER_OP;

    // Called by server to request shared resources be created for a client resource
    typedef HRESULT (__cdecl *PFNREQUESTCLIENTRESOURCE)(HANSELCLIENT hAnselClient,
                                                        HCLIENTRESOURCE hClientResource,
                                                        void * pToServerPrivateData,
                                                        void * pToClientPrivateData);
    
    // Called by server to copy client resource data to the shared resource
    typedef HRESULT (__cdecl *PFNCOPYCLIENTRESOURCE)   (HANSELCLIENT hAnselClient,
                                                        HCLIENTRESOURCE hClientResource,
                                                        ANSEL_TRANSFER_OP op,
                                                        DWORD subResourceIndex,
                                                        DWORD acquireKey,
                                                        DWORD releaseKey);

    // If pRect is NULL, the client is expected to do a full resource copy
    typedef HRESULT (__cdecl *PFNCOPYCLIENTRESOURCEWITHRECT)   (HANSELCLIENT hAnselClient,
                                                                HCLIENTRESOURCE hClientResource,
                                                                ANSEL_TRANSFER_OP op,
                                                                DWORD subResourceIndex,
                                                                DWORD acquireKey,
                                                                DWORD releaseKey,
                                                                const RECT * pRect);

    // Called by server to copy server resource data to the shared resource
    typedef HRESULT (__cdecl *PFNSENDCLIENTRESOURCE)   (HANSELCLIENT hAnselClient,
                                                        HCLIENTRESOURCE hClientResource,
                                                        ANSEL_TRANSFER_OP op,
                                                        DWORD subResourceIndex,
                                                        DWORD acquireKey,
                                                        DWORD releaseKey);
    
    // If pRect is NULL, the client is expected to do a full resource copy
    typedef HRESULT (__cdecl *PFNSENDCLIENTRESOURCEWITHRECT)   (HANSELCLIENT hAnselClient,
                                                                HCLIENTRESOURCE hClientResource,
                                                                ANSEL_TRANSFER_OP op,
                                                                DWORD subResourceIndex,
                                                                DWORD acquireKey,
                                                                DWORD releaseKey,
                                                                const RECT * pRect);
    
    // Called by server to copy client resource data to the shared resource
    typedef HRESULT (__cdecl *PFNCOPYCLIENTRESOURCE12) (HCMDLIST hCmdList,
                                                        HANSELCLIENT hAnselClient,
                                                        HCLIENTRESOURCE hClientResource,
                                                        ANSEL_TRANSFER_OP op,
                                                        DWORD subResourceIndex,
                                                        DWORD acquireKey,
                                                        DWORD releaseKey);
    
    // If pRect is NULL, the client is expected to do a full resource copy
    typedef HRESULT (__cdecl *PFNCOPYCLIENTRESOURCEWITHRECT12) (HCMDLIST hCmdList,
                                                                HANSELCLIENT hAnselClient,
                                                                HCLIENTRESOURCE hClientResource,
                                                                ANSEL_TRANSFER_OP op,
                                                                DWORD subResourceIndex,
                                                                DWORD acquireKey,
                                                                DWORD releaseKey,
                                                                const RECT * pRect);

    // Called by server to copy server resource data to the shared resource
    typedef HRESULT (__cdecl *PFNSENDCLIENTRESOURCE12) (HCMDLIST hCmdList,
                                                        HANSELCLIENT hAnselClient,
                                                        HCLIENTRESOURCE hClientResource,
                                                        ANSEL_TRANSFER_OP op,
                                                        DWORD subResourceIndex,
                                                        DWORD acquireKey,
                                                        DWORD releaseKey);

    // If pRect is NULL, the client is expected to do a full resource copy
    typedef HRESULT (__cdecl *PFNSENDCLIENTRESOURCEWITHRECT12) (HCMDLIST hCmdList,
                                                                HANSELCLIENT hAnselClient,
                                                                HCLIENTRESOURCE hClientResource,
                                                                ANSEL_TRANSFER_OP op,
                                                                DWORD subResourceIndex,
                                                                DWORD acquireKey,
                                                                DWORD releaseKey,
                                                                const RECT * pRect);

    typedef HRESULT (__cdecl *PFNGETDEVICESTATES)      (HANSELCLIENT hAnselClient,
                                                        AnselDeviceStates * pDeviceState);
    typedef HRESULT (__cdecl *PFNGETDEVICESTATES12)    (HANSELCLIENT hAnselClient,
                                                        HCMDLIST hCmdList,
                                                        AnselDeviceStates * pDeviceState);
    struct AnselClientResourceInfo
    {
        DWORD Width;
        DWORD Height;
        DWORD Depth;
        DWORD Format;
        DWORD SampleCount;
        DWORD SampleQuality;
        DWORD SubresourceCount;
    };

    typedef HRESULT(__cdecl *PFNGETCLIENTRESOURCEINFO)(HANSELCLIENT hAnselClient,
        HCLIENTRESOURCE hClientResource,
        AnselClientResourceInfo * pResourceinfo);

    typedef HRESULT(__cdecl *PFNDISABLECLIENT)(HANSELCLIENT hAnselClient);

    typedef HRESULT(__cdecl *PFNENTERLIGHTWEIGHTMODE) (HANSELCLIENT hAnselClient);

    typedef HRESULT(__cdecl *PFNEXITLIGHTWEIGHTMODE) (HANSELCLIENT hAnselClient);

    struct ClientFunctionTable
    {
        PFNREQUESTCLIENTRESOURCE RequestClientResource;
        PFNCOPYCLIENTRESOURCE CopyClientResource;
        PFNSENDCLIENTRESOURCE SendClientResource;
        PFNGETDEVICESTATES GetDeviceStates;
        PFNGETCLIENTRESOURCEINFO GetClientResourceInfo;
        PFNDISABLECLIENT DisableClient;
        PFNCOPYCLIENTRESOURCE12 CopyClientResource12;
        PFNSENDCLIENTRESOURCE12 SendClientResource12;
        PFNGETDEVICESTATES12 GetDeviceStates12;

        // Only use if Ansel server reports version 5.1 or later
        PFNCOPYCLIENTRESOURCEWITHRECT CopyClientResourceWithRect;
        PFNSENDCLIENTRESOURCEWITHRECT SendClientResourceWithRect;
        PFNCOPYCLIENTRESOURCEWITHRECT12 CopyClientResourceWithRect12;
        PFNSENDCLIENTRESOURCEWITHRECT12 SendClientResourceWithRect12;

        // Only use if Ansel server reports version 6.1 or later
        PFNENTERLIGHTWEIGHTMODE EnterLightweightMode;
        PFNEXITLIGHTWEIGHTMODE ExitLightweightMode;
    };
}

#endif //ANSEL_VERSION_ONLY

#endif //_ANSEL_H_
