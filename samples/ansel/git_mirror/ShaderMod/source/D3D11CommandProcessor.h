#pragma once

#include "ir/TypeEnums.h"

#include <assert.h>

namespace shadermod
{

    enum class CmdProcSystemConst: unsigned int
    {
        kDT = 0,            // float
        kElapsedTime,       // float
        kFrame,             // int
        kScreenSize,        // float, dims=2
        kCaptureState,      // int
        kTileUV,            // float, dims=4
        kDepthAvailable,    // int
        kHDRAvailable,      // int
        kHUDlessAvailable,  // int
        kUnkown,
        kNUM_ENTRIES
    };

    enum class CmdProcConstDataType
    {
        kBool = 0,
        kInt,
        kUInt,
        kFloat,

        // Elements like Float2/Float4 are not needed because there is additional parameter - dimensionality

        kNUM_ENTRIES
    };

    static CmdProcConstDataType getCmdProcSystemConstElementDataType(const CmdProcSystemConst& sc)
    {
        switch (sc)
        {
        case CmdProcSystemConst::kDT:
            return CmdProcConstDataType::kFloat;
        case CmdProcSystemConst::kElapsedTime:
            return CmdProcConstDataType::kFloat;
        case CmdProcSystemConst::kFrame:
            return CmdProcConstDataType::kInt;
        case CmdProcSystemConst::kScreenSize:
            return CmdProcConstDataType::kFloat;
        case CmdProcSystemConst::kCaptureState:
            return CmdProcConstDataType::kInt;
        case CmdProcSystemConst::kTileUV:
            return CmdProcConstDataType::kFloat;
        case CmdProcSystemConst::kDepthAvailable:
            return CmdProcConstDataType::kInt;
        case CmdProcSystemConst::kHDRAvailable:
            return CmdProcConstDataType::kInt;
        case CmdProcSystemConst::kHUDlessAvailable:
            return CmdProcConstDataType::kInt;
        default:
            return CmdProcConstDataType::kNUM_ENTRIES;
        }
    }

    static unsigned int getCmdProcSystemConstDimensions(const CmdProcSystemConst& sc)
    {
        switch (sc)
        {
        case CmdProcSystemConst::kDT:
            return 1;
        case CmdProcSystemConst::kElapsedTime:
            return 1;
        case CmdProcSystemConst::kFrame:
            return 1;
        case CmdProcSystemConst::kScreenSize:
            return 2;
        case CmdProcSystemConst::kCaptureState:
            return 1;
        case CmdProcSystemConst::kTileUV:
            return 4;
        case CmdProcSystemConst::kDepthAvailable:
            return 1;
        case CmdProcSystemConst::kHDRAvailable:
            return 1;
        case CmdProcSystemConst::kHUDlessAvailable:
            return 1;
        default:
            return 1;
        }
    }

    static size_t getCmdProcConstDataElementTypeSize(const CmdProcConstDataType& t)
    {
        switch (t)
        {
        case CmdProcConstDataType::kBool:
            return sizeof(ir::userConstTypes::Bool);
        case CmdProcConstDataType::kInt:
            return sizeof(ir::userConstTypes::Int);
        case CmdProcConstDataType::kUInt:
            return sizeof(ir::userConstTypes::UInt);
        case CmdProcConstDataType::kFloat:
            return sizeof(ir::userConstTypes::Float);
        default:
            return 0;
        }
    }

    typedef unsigned int CmdProcConstHandle;

    static bool isSystemConst(CmdProcConstHandle h)
    {
        return (h < (unsigned int)CmdProcSystemConst::kNUM_ENTRIES);
    }

    static CmdProcSystemConst toSystemConst(CmdProcConstHandle h)
    {
        return (h < (unsigned int)CmdProcSystemConst::kNUM_ENTRIES) ? (CmdProcSystemConst)h : CmdProcSystemConst::kNUM_ENTRIES;
    }

    static unsigned int toUserConstIndex(CmdProcConstHandle h)
    {
        return (h < (unsigned int)CmdProcSystemConst::kNUM_ENTRIES) ? 0xFFffFFff : (h - (unsigned int)CmdProcSystemConst::kNUM_ENTRIES);
    }

    static CmdProcConstHandle makeCmdProcConstHandle(CmdProcSystemConst sc)
    {
        assert(sc < CmdProcSystemConst::kNUM_ENTRIES);

        return (CmdProcConstHandle) sc;
    }

    static CmdProcConstHandle makeCmdProcConstHandle(unsigned int userConstIndex)
    {
        return (CmdProcConstHandle)((unsigned int)CmdProcSystemConst::kNUM_ENTRIES + userConstIndex);
    }
        
    //struct CmdProcTextureDesc
    //{
    //  ID3D11Texture2D * pTexObject;
    //  int slot;
    //};

    struct CmdProcSamplerDesc
    {
        ID3D11SamplerState * pSampler;
        int slot;
    };

    struct CmdProcConstantDesc
    {
        CmdProcConstHandle constHandle;
        int offsetInComponents;
    };

    class CmdProcConstantBufDesc
    {
    public:

        CmdProcConstantBufDesc(ID3D11Buffer * pBuffer, int slot):
            m_pBuffer(pBuffer),
            m_slot(slot)
        {
        }

        std::vector<CmdProcConstantDesc> m_constants;
        ID3D11Buffer * m_pBuffer;
        int m_slot;

    protected:
    };

    enum class ShaderResourceKind : unsigned int
    {
        // 0 - 0x8000000 reserved for future use - possibly user SRV ids
        kSystemResourceBase = 0x8000000,
        kColor,
        kDepth,
        kHDR,
        kHUDless,
        kColorBase
    };

    struct CmdProcShaderResourceDesc
    {
        ID3D11ShaderResourceView * pResource;
        int slot;
        ShaderResourceKind kind;
    };

    class CmdProcPass
    {
    public:

        ID3D11VertexShader *                    m_vertexShader          = nullptr;
        ID3D11PixelShader *                     m_pixelShader           = nullptr;

        ID3D11RasterizerState *                 m_rasterizerState;
        ID3D11DepthStencilState *               m_depthStencilState;
        ID3D11BlendState *                      m_alphaBlendState;

        //vector<CmdProcTextureDesc>            m_textureDescs;
        std::vector<CmdProcSamplerDesc>         m_samplerDescs;
        std::vector<CmdProcConstantBufDesc>     m_constantBufPSDescs;
        std::vector<CmdProcConstantBufDesc>     m_constantBufVSDescs;
        std::vector<CmdProcShaderResourceDesc>  m_shaderResourceDescs;
        std::vector<ID3D11RenderTargetView *>   m_renderTargets;

        float                                   m_width = -1.0f, m_height = -1.0f;

    protected:
    };
    
    class CmdProcEffect
    {
    public:

        ID3D11Texture2D * GetInputColorTex() const
        {
            if (m_passes.empty() || m_passes[0].m_shaderResourceDescs.empty()) return NULL;
            ID3D11Resource * inputColorTex = NULL;
            m_passes[0].m_shaderResourceDescs[0].pResource->GetResource(&inputColorTex);
            return static_cast<ID3D11Texture2D *>(inputColorTex);
        }
        ID3D11Texture2D * GetOutputColorTex() const
        {
            if (m_passes.empty() || m_passes[0].m_renderTargets.empty()) return NULL;
            ID3D11Resource * outputColorTex = NULL;
            m_passes[m_passes.size()-1].m_renderTargets[0]->GetResource(&outputColorTex);
            return static_cast<ID3D11Texture2D *>(outputColorTex);
        }

        ID3D11Texture2D *           m_inputColorTex     = nullptr;
        ID3D11Texture2D *           m_inputDepthTex     = nullptr;
        ID3D11Texture2D *           m_outputColorTex    = nullptr;

        std::vector<CmdProcPass>    m_passes;
    
        void destroy()
        {
            m_inputColorTex = nullptr;
            m_inputDepthTex = nullptr;
            m_outputColorTex = nullptr;
            m_passes.clear();
        }

    protected:

    };

}
