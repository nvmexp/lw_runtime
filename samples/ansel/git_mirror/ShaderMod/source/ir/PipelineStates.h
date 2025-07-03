#pragma once

namespace shadermod
{
namespace ir
{

    enum class RasterizerFillMode
    {
        // Based on D3D11_FILL_MODE 

        kWireframe = 2,
        kSolid = 3,

        kNUM_ENTRIES
    };

    enum class RasterizerLwllMode
    {
        // Based on D3D11_LWLL_MODE

        kNone = 1,
        kFront = 2,
        kBack = 3,

        kNUM_ENTRIES
    };

    struct RasterizerState
    {
        // Based on D3D11_RASTERIZER_DESC

        RasterizerFillMode fillMode;
        RasterizerLwllMode lwllMode;

        int32_t depthBias;
        float depthBiasClamp;
        float slopeScaledDepthBias;

        uint8_t frontCounterClockwise;
        uint8_t depthClipEnable;
        uint8_t scissorEnable;
        uint8_t multisampleEnable;
        uint8_t antialiasedLineEnable;
    };

    enum class DepthWriteMask
    {
        // Based on D3D11_DEPTH_WRITE_MASK

        kZero = 0,
        kAll = 1,

        kNUM_ENTRIES
    };

    enum class ComparisonFunc
    {
        // Based on D3D11_COMPARISON_FUNC

        kNever = 1,
        kLess = 2,
        kEqual = 3,
        kLessEqual = 4,
        kGreater = 5,
        kNotEqual = 6,
        kGreaterEqual = 7,
        kAlways = 8,

        kNUM_ENTRIES
    };

    enum class StencilOp
    {
        // Based on D3D11_STENCIL_OP

        kKeep = 1,
        kZero = 2,
        kReplace = 3,
        kIncrSat = 4,
        kDecrSat = 5,
        kIlwert = 6,
        kIncr = 7,
        kDecr = 8,

        kNUM_ENTRIES
    };

    struct DepthStencilOp
    {
        // Based on D3D11_DEPTH_STENCILOP_DESC

        StencilOp failOp;
        StencilOp depthFailOp;
        StencilOp passOp;
        ComparisonFunc func;
    };

    struct DepthStencilState
    {
        static const uint8_t defaultStencilReadMask = 0xFF;    //D3D11_DEFAULT_STENCIL_READ_MASK
        static const uint8_t defaultStencilWriteMask = 0xFF;  //D3D11_DEFAULT_STENCIL_WRITE_MASK

        // Based on D3D11_DEPTH_STENCIL_DESC

        DepthStencilOp frontFace;
        DepthStencilOp backFace;

        DepthWriteMask depthWriteMask;
        ComparisonFunc depthFunc;

        uint8_t stencilReadMask;
        uint8_t stencilWriteMask;
        uint8_t isDepthEnabled;
        uint8_t isStencilEnabled;
    };

    enum class BlendCoef
    {
        // Based on D3D11_BLEND

        kZero = 1,
        kOne = 2,
        kSrcColor = 3,
        kIlwSrcColor = 4,
        kSrcAlpha = 5,
        kIlwSrcAlpha = 6,
        kDstAlpha = 7,
        kIlwDstAlpha = 8,
        kDstColor = 9,
        kIlwDstColor = 10,
        kSrcAlphaSat = 11,
        kBlendFactor = 14,
        kIlwBlendFactor = 15,
        kSrc1Color = 16,
        kIlwSrc1Color = 17,
        kSrc1Alpha = 18,
        kIlwSrc1Alpha = 19,

        kNUM_ENTRIES
    };

    enum class BlendOp
    {
        // Based on D3D11_BLEND_OP

        kAdd = 1,
        kSub = 2,
        kRevSub = 3,
        kMin = 4,
        kMax = 5,

        kNUM_ENTRIES
    };

    enum class ColorWriteEnableBits
    {
        // Based on D3D11_COLOR_WRITE_ENABLE

        kRed = 1,
        kGreen = 2,
        kBlue = 4,
        kAlpha = 8,

        kAll = ( ((kRed|kGreen)  | kBlue)  | kAlpha )
    };

    struct AlphaBlendRenderTargetState
    {
        // Based on D3D11_RENDER_TARGET_BLEND_DESC

        BlendCoef src;
        BlendCoef dst;
        BlendOp op;

        BlendCoef srcAlpha;
        BlendCoef dstAlpha;
        BlendOp opAlpha;

        ColorWriteEnableBits renderTargetWriteMask;
        uint8_t isEnabled;
    };

    struct AlphaBlendState
    {
        static const uint8_t renderTargetsNum = 8;

        // Based on D3D11_BLEND_DESC

        AlphaBlendRenderTargetState renderTargetState[ renderTargetsNum ];

        uint8_t alphaToCoverageEnable;
        uint8_t independentBlendEnable;
    };

    static
    void initRasterizerState(RasterizerState * rasterizerState)
    {
        if (rasterizerState == nullptr)
            return;

        RasterizerState & rs = *rasterizerState;

        // Defaults from D3D11_RASTERIZER_DESC
        rs.fillMode = RasterizerFillMode::kSolid;
        rs.lwllMode = RasterizerLwllMode::kBack;
        rs.frontCounterClockwise = 0;
        rs.depthBias = 0;
        rs.slopeScaledDepthBias = 0.0f;
        rs.depthBiasClamp = 0.0f;
        rs.depthClipEnable = 1;
        rs.scissorEnable = 0;
        rs.multisampleEnable = 0;
        rs.antialiasedLineEnable = 0;
    }

    static
    void initDepthStencilState(DepthStencilState * depthStencilState)
    {
        if (depthStencilState == nullptr)
            return;

        DepthStencilState & ds = *depthStencilState;

        // Defaults from D3D11_DEPTH_STENCIL_DESC
        ds.isDepthEnabled = 1;
        ds.depthWriteMask = DepthWriteMask::kAll;
        ds.depthFunc = ComparisonFunc::kLess;
        ds.isStencilEnabled = 0;
        ds.stencilReadMask = DepthStencilState::defaultStencilReadMask;
        ds.stencilWriteMask = DepthStencilState::defaultStencilWriteMask;
        ds.frontFace.func = ComparisonFunc::kAlways;
        ds.backFace.func = ComparisonFunc::kAlways;
        ds.frontFace.depthFailOp = StencilOp::kKeep;
        ds.backFace.depthFailOp = StencilOp::kKeep;
        ds.frontFace.passOp = StencilOp::kKeep;
        ds.backFace.passOp = StencilOp::kKeep;
        ds.frontFace.failOp = StencilOp::kKeep;
        ds.backFace.failOp = StencilOp::kKeep;
    }

    static
    void initAlphaBlendState(AlphaBlendState * alphaBlendState)
    {
        if (alphaBlendState == nullptr)
            return;

        AlphaBlendState & as = *alphaBlendState;

        // Defaults from D3D11_BLEND_DESC
        as.alphaToCoverageEnable = 0;
        as.independentBlendEnable = 0;

        for (uint8_t idx = 0; idx < AlphaBlendState::renderTargetsNum; ++idx)
        {
            AlphaBlendRenderTargetState & asrts = as.renderTargetState[idx];
            asrts.isEnabled = 0;

            asrts.src = BlendCoef::kOne;
            asrts.dst = BlendCoef::kZero;
            asrts.op = BlendOp::kAdd;

            asrts.srcAlpha = BlendCoef::kOne;
            asrts.dstAlpha = BlendCoef::kZero;
            asrts.opAlpha = BlendOp::kAdd;

            asrts.renderTargetWriteMask = ColorWriteEnableBits::kAll;
        }
    }

}
}