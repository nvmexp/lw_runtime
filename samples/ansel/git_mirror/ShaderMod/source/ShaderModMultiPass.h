#pragma once

#include "Timer.h"
#include "ResourceManager.h"
#include "ir/Effect.h"
#include "MultipassConfigParserError.h"
#include "frameworks/lwbin/ngx_lwbin.h"
#include "Hash.h"

#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <vector>

#define SHADER_CAPTURE_NOT_STARTED          0
#define SHADER_CAPTURE_REGULAR              1
#define SHADER_CAPTURE_REGULARSTEREO        2
#define SHADER_CAPTURE_HIGHRES              3
#define SHADER_CAPTURE_360                  4
#define SHADER_CAPTURE_360STEREO            5

bool isHdrFormatSupported(DXGI_FORMAT format);

struct UAVSupport
{
    bool D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW = false;
    bool D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD = false;
    bool D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE = false;
    bool UAVSupported() const { return D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW && D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD && D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE; }
};

namespace shadermod
{
    class MultiPassEffect
    {
    public:
        
        MultiPassEffect(
            const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,
            const std::map<std::wstring, std::wstring> & fxExtensionToolMap,

            const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
            const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
            const ir::Effect::InputData & colorBaseInput,

            ID3D11Device* d3dDevice, D3DCompilerHandler* d3dCompiler,

            const std::set<Hash::Effects> * pExpectedHashSet,
            bool compareHashes
            );
        MultiPassEffect(
            const wchar_t * fxToolFilepath, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,

            const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
            const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
            const ir::Effect::InputData & colorBaseInput,

            ID3D11Device* d3dDevice, D3DCompilerHandler* d3dCompiler,

            const std::set<Hash::Effects> * pExpectedHashSet,
            bool compareHashes
            );

        ~MultiPassEffect();

        void initializeEffect(
            const wchar_t * fxToolFilepath, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,
            const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
            const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
            const ir::Effect::InputData & colorBaseInput,
            const std::set<Hash::Effects> * pExpectedHashSet,
            bool compareHashes
            );

        const ir::UserConstantManager& getUserConstantManager() const;
        ir::UserConstantManager& getUserConstantManager();
        bool isForceUpdateOfSystemConstantsNeeded() const;
        void markSystemConstantsUpdatedOnce();
        ir::FragmentFormat  getOutFormat() const;
        ID3D11Texture2D* getOutputColorTexture() const;
        const std::wstring& getFxToolFilepath() const;
        const std::wstring& getRootDir() const;
        const std::wstring& getTempsDir() const;
        const std::wstring& getFxFilename() const;
        const std::wstring& getFxFileFullPath() const;
        unsigned int getOutputWidth() const;
        unsigned int getOutputHeight() const;
        const CmdProcEffect& getLowLevelEffect() const;

        std::set<Hash::Effects> m_expectedHashSet;
        Hash::Effects m_callwlatedHashes;

        const std::set<Hash::Effects>& MultiPassEffect::getExpectedHashSet() const { return m_expectedHashSet; }
        const Hash::Effects& MultiPassEffect::getCallwlatedHashes() const { return m_callwlatedHashes; }

        const uint8_t * MultiPassEffect::getCallwlatedShaderHash() const;
        const uint8_t * MultiPassEffect::getCallwlatedResourceHash() const;
        const uint8_t * MultiPassEffect::getCallwlatedACEFBinaryHash() const;

        bool isDepthRequired() const { return m_effectRequiresDepth; }
        bool isHUDlessRequired() const { return m_effectRequiresHUDless; }
        bool isHDRRequired() const { return m_effectRequiresHDR; }

        bool changeInputs(
            const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
            const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
            const ir::Effect::InputData & colorBaseInput,
            bool ignoreDepthTextureNotSet
            );

    private:

        bool                m_systemConstantsNeverSet;
        bool                m_effectRequiresDepth = false;
        bool                m_effectRequiresHUDless = false;
        bool                m_effectRequiresHDR = false;
        bool                m_effectRequiresColorBase = false;

        CmdProcEffect       m_effect;
        ResourceManager     m_resourceManager;
        ir::Effect          m_irEffect;
    
        ir::FragmentFormat  m_outputFormat = ir::FragmentFormat::kNUM_ENTRIES;
        unsigned int        m_outputWidth = 0xFFffFFff;
        unsigned int        m_outputHeight = 0xFFffFFff;
        
        std::wstring        m_fxToolFilepath;   //TODO do we really need to store it here?
        std::wstring        m_rootDir;          //TODO do we really need to store it here?
        std::wstring        m_tempsDir;         //TODO do we really need to store it here?
        std::wstring        m_fxFilename;
        std::wstring        m_fxFilepath;
    };


    class MultiPassProcessor
    {
    public:
        static const unsigned int undefinedSize = 0xFFffFFff;

        MultiPassProcessor(const MultiPassProcessor&) = delete;
        MultiPassProcessor& operator=(const MultiPassProcessor&) = delete;

        MultiPassProcessor(const wchar_t * compilerPath,  bool doInitFramework = true, LARGE_INTEGER adapterLUID = { 0 });
        MultiPassProcessor(bool doInitFramework = true, LARGE_INTEGER adapterLUID = { 0 });

        virtual ~MultiPassProcessor();

        bool initFramework(LARGE_INTEGER adapterLUID = { 0 });

        bool isDeviceValid() const
        {
            return m_isValid;
        }
                    
        unsigned int getWidth() const { return m_width; }
        unsigned int getHeight() const { return m_height; }
        unsigned int getDepthWidth() const { return m_depthWidth; }
        unsigned int getDepthHeight() const { return m_depthHeight; }

        void destroyDevice();
        
        void setD3DCompilerPath(const wchar_t * compilerPath);
        
        unsigned int getNumEffects() const;
        MultiPassEffect* getEffect(unsigned int idx);
        MultiPassEffect* getEffect(unsigned int idx) const;

        MultipassConfigParserError relinkEffects(bool ignoreTexturesNotSet);

        MultipassConfigParserError replaceEffect(
                    const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir,
                    const std::map<std::wstring, std::wstring> & fxExtensionToolMap,
                    const wchar_t * filename_yaml, MultiPassEffect ** effectPtr, int stackIdx,
                    const std::set<Hash::Effects> * pExpectedHashSet,
                    bool compareHashes
                    );

        MultipassConfigParserError removeSingleEffect(unsigned int idx, bool dontDestroy = false);

        MultipassConfigParserError pushBackEffect(
                    const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir,
                    const std::map<std::wstring, std::wstring> & fxExtensionToolMap,
                    const wchar_t * filename_yaml, MultiPassEffect ** effectPtr,
                    const std::set<Hash::Effects> * pExpectedHashSet,
                    bool compareHashes
                    );
        void popBackEffect(bool dontDestroy = false);

#if 0 //better not use those methods as they woul leave your stak in a dangling state if an effect down the pipe fails the compilation
        MultipassConfigParserError insertEffect(MultiPassEffect* eff, unsigned int idx);
        MultipassConfigParserError eraseEffect(unsigned int idx, bool dontDestroy = false);
#endif
        //it's the user's duty to destroy all effects that were created and aren't in the stack - no automatic cleanup here!
        void destroyAllEffectsInStack();
        
        bool MultiPassProcessor::copyData(ID3D11ShaderResourceView * source, ID3D11RenderTargetView* dest, float width, float height, bool skipIAVSSetup = false);
        bool processData(ID3D11RenderTargetView* dest);

        bool isDepthRequiredOnStack() const;
        bool isHUDlessRequiredOnStack() const;
        bool isHDRRequiredOnStack() const;

        //returns true if effect rebuild is needed
        bool setInputs(
                    const ir::Effect::InputData &   finalColorInput,
                    const ir::Effect::InputData &   depthInput,
                    const ir::Effect::InputData &   hudlessInput,
                    const ir::Effect::InputData &   hdrInput,
                    MultipassConfigParserError& err
                    );

        //returns true if effect rebuild is needed
        // TODO avoroshilov: this function seems obsolete?
        bool setWidthHeight(
                    unsigned int        width,
                    unsigned int        height,
                    unsigned int        depthWidth,
                    unsigned int        depthHeight,
                    unsigned int        hudlessWidth,
                    unsigned int        hudlessHeight,
                    unsigned int        hdrWidth,
                    unsigned int        hdrHeight,
                    MultipassConfigParserError& err
                    );

        bool areInputsValid() const;

        struct TileInfo
        {
            float   m_tileTLU = 0.0f,
                    m_tileTLV = 0.0f,
                    m_tileBRU = 1.0f,
                    m_tileBRV = 1.0f;
        };
        
        TileInfo& getTileInfo(void)
        {
            return m_tileInfo;
        }
        void setTileInfo(const TileInfo& info)
        {
            m_tileInfo = info;
        }

        void setCaptureState(int cs)
        {
            m_captureState = cs;
        }
        int getCaptureState() const
        {
            return m_captureState;
        }

        void setShaderBuffersAvailability(bool isDepthAvailableShader, bool isHDRAvailableShader, bool isHUDlessAvailableShader)
        {
            m_isDepthAvailableShader = isDepthAvailableShader;
            m_isHDRAvailableShader = isHDRAvailableShader;
            m_isHUDlessAvailableShader = isHUDlessAvailableShader;
        }

        bool isDefunctEffectOnStack() const
        {
            return m_isDefunctEffectInStack;
        }

        D3DCompilerHandler & getD3DCompiler()
        {
            return m_d3dCompiler;
        }

    protected:

        D3DCompilerHandler      m_d3dCompiler;
        ID3D11Device*           m_d3dDevice = nullptr;
        ID3D11DeviceContext*    m_immediateContext = nullptr;
        bool                    m_isValid = false;
    
    private:
        
        MultiPassEffect* createEffectFromACEF(
                    const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * filename_yaml,

                    const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
                    const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
                    const ir::Effect::InputData & colorBaseInput,

                    MultipassConfigParserError& err,

                    const std::set<Hash::Effects> * pExpectedHashSet,
                    bool compareHashes
                    );

        MultiPassEffect* MultiPassProcessor::createEffectFromACEF(
                    const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,
                    const std::map<std::wstring, std::wstring> & fxExtensionToolMap,

                    const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
                    const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
                    const ir::Effect::InputData & colorBaseInput,

                    MultipassConfigParserError& err,

                    const std::set<Hash::Effects> * pExpectedHashSet,
                    bool compareHashes
                    );

        void destroyEffect(MultiPassEffect* eff);

        void pushBackEffect(MultiPassEffect* eff);

        void onLastEffectUpdated(bool updateSucceeded);
    
        CmdProcConstDataType    getConstantDataType(CmdProcConstHandle h, const MultiPassEffect *eff) const;
        unsigned int            getConstantDataDimensions(CmdProcConstHandle h, const MultiPassEffect *eff) const;

        bool                    writeConstantValue(CmdProcConstHandle h, const MultiPassEffect *eff, void* buf, size_t bytesToCopy) const;
        bool                    isConstantDirty(CmdProcConstHandle h, const MultiPassEffect *eff) const;
        void                    markConstantClean(CmdProcConstHandle h, MultiPassEffect *eff);
            
        bool createCopyShader();
        void destroyCopyShader();

        bool updateOutSRV();
        void destroyOutSRV();

        MultipassConfigParserError rebuildEffectsInStack(unsigned int startFrom, bool compareHashes);

        void CheckLwbinSupportOnHW(ID3D11Device *InDevice);
            
        TileInfo                m_tileInfo;
        int                     m_captureState = 0;
        int                     m_processingFrameNum = 0;

        bool                    m_isDepthAvailableShader = false;
        bool                    m_isHDRAvailableShader = false;
        bool                    m_isHUDlessAvailableShader = false;

        IDXGIAdapter1 *         m_pAdapter = nullptr;
        D3D_DRIVER_TYPE         m_driverType = D3D_DRIVER_TYPE_NULL;
        D3D_FEATURE_LEVEL       m_featureLevel = D3D_FEATURE_LEVEL_11_0;

        ID3D11VertexShader*     m_vertexShader = nullptr;

        ir::Pool<MultiPassEffect, 3>    m_effectsPool;
        std::vector<MultiPassEffect*>   m_effectsStackInternal;

        Timer m_timer;

        double                  m_dt = 0.0;
        double                  m_elapsedTime = 0.0;

        std::vector<CmdProcConstHandle> m_referencedConstantsCache;

        ID3D11ShaderResourceView*   m_copyInputSRV = nullptr;

        ID3D11PixelShader*          m_copyPixelShader = nullptr;
        ID3D11SamplerState*         m_samplerLinear = nullptr;

        bool                    m_isDefunctEffectInStack = false;
        
        ID3D11Texture2D*        m_colorSourceTexture = nullptr;
        ID3D11Texture2D*        m_depthSourceTexture = nullptr;
        ID3D11Texture2D*        m_hudlessSourceTexture = nullptr;
        ID3D11Texture2D*        m_hdrSourceTexture = nullptr;
        ir::FragmentFormat      m_colorSourceTextureFormat = ir::FragmentFormat::kNUM_ENTRIES;
        ir::FragmentFormat      m_depthSourceTextureFormat = ir::FragmentFormat::kNUM_ENTRIES;
        ir::FragmentFormat      m_hudlessSourceTextureFormat = ir::FragmentFormat::kNUM_ENTRIES;
        ir::FragmentFormat      m_hdrSourceTextureFormat = ir::FragmentFormat::kNUM_ENTRIES;
        unsigned int            m_width = undefinedSize;
        unsigned int            m_height = undefinedSize;
        unsigned int            m_depthWidth = undefinedSize;
        unsigned int            m_depthHeight = undefinedSize;
        unsigned int            m_hudlessWidth = undefinedSize;
        unsigned int            m_hudlessHeight = undefinedSize;
        unsigned int            m_hdrWidth = undefinedSize;
        unsigned int            m_hdrHeight = undefinedSize;

    protected:
        // Lwbin Data
        HRESULT InitLwbin();
        HRESULT ShutdownLwbin();
        HRESULT ColwertToUAVSupportedFormat(DXGI_FORMAT& format);
        UAVSupport GetUAVFormatSupport(DXGI_FORMAT format);
        void LogUAVFormatSupport(DXGI_FORMAT format, const UAVSupport& support);
        HRESULT RunLwbin(ID3D11Resource *Input, ID3D11Resource *Output, UINT Width, UINT Height);

        NGXLwbin *m_LwbinAPI = nullptr;
        NGXLwbinKernel m_SharpenKernel_HDR = {};
        NGXLwbinKernel m_SharpenKernel_LDR = {};
        NGXLwbinParameters *m_SharpenParams = {};
        std::unordered_map<DXGI_FORMAT, UAVSupport> m_uavFormatSupport;

        // Sharpen
        struct SharpenLwbinData
        {
            float m_SharpenParameter;
            float m_DenoiseParameter;
            bool m_HWSupport;
            unsigned char* m_Kernel_LDR;
            unsigned int m_KernelSizeInBytes_LDR;
            unsigned char* m_Kernel_HDR;
            unsigned int m_KernelSizeInBytes_HDR;
        };
        SharpenLwbinData m_SharpenLwbinData;
        bool m_HWSupportsLwbin = false;
    };
}
