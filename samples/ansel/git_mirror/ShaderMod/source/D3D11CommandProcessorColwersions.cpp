#pragma once

#include <assert.h>
#include <d3d11.h>

#include "D3D11CommandProcessorColwersions.h"

namespace shadermod
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Rasterizer State
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    D3D11_FILL_MODE colwertRasterFillModeD3D11(const ir::RasterizerFillMode & irRasterFillMode)
    {
        switch (irRasterFillMode)
        {
        case ir::RasterizerFillMode::kSolid:
            return D3D11_FILL_SOLID;
        case ir::RasterizerFillMode::kWireframe:
            return D3D11_FILL_WIREFRAME;
        default:
            {
                assert(false && "Failed to colwert D3D11_FILL_MODE");
                return D3D11_FILL_SOLID;
            }
        }
    };

    D3D11_LWLL_MODE colwertRasterLwllModeD3D11(const ir::RasterizerLwllMode & irRasterLwllMode)
    {
        switch (irRasterLwllMode)
        {
        case ir::RasterizerLwllMode::kBack:
            return D3D11_LWLL_BACK;
        case ir::RasterizerLwllMode::kFront:
            return D3D11_LWLL_FRONT;
        case ir::RasterizerLwllMode::kNone:
            return D3D11_LWLL_NONE;
        default:
            {
                assert(false && "Failed to colwert D3D11_LWLL_MODE");
                return D3D11_LWLL_NONE;
            }
        }
    };

    D3D11_RASTERIZER_DESC CmdProcColwertRasterizerStateD3D11(const ir::RasterizerState & irRasterState)
    {
        D3D11_RASTERIZER_DESC d3dRasterState;

        d3dRasterState.FillMode = colwertRasterFillModeD3D11(irRasterState.fillMode);
        d3dRasterState.LwllMode = colwertRasterLwllModeD3D11(irRasterState.lwllMode);

        d3dRasterState.DepthBias = irRasterState.depthBias;
        d3dRasterState.DepthBiasClamp = irRasterState.depthBiasClamp;
        d3dRasterState.SlopeScaledDepthBias = irRasterState.slopeScaledDepthBias;

        d3dRasterState.FrontCounterClockwise = irRasterState.frontCounterClockwise;
        d3dRasterState.DepthClipEnable = irRasterState.depthClipEnable;
        d3dRasterState.ScissorEnable = irRasterState.scissorEnable;
        d3dRasterState.MultisampleEnable = irRasterState.multisampleEnable;
        d3dRasterState.AntialiasedLineEnable = irRasterState.antialiasedLineEnable;

        return d3dRasterState;
    };



    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // DepthStencil State
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    D3D11_COMPARISON_FUNC colwertComparisonFuncD3D11(const ir::ComparisonFunc & irComparisonFunc)
    {
        switch (irComparisonFunc)
        {
        case ir::ComparisonFunc::kAlways:
            return D3D11_COMPARISON_ALWAYS;
        case ir::ComparisonFunc::kEqual:
            return D3D11_COMPARISON_EQUAL;
        case ir::ComparisonFunc::kGreater:
            return D3D11_COMPARISON_GREATER;
        case ir::ComparisonFunc::kGreaterEqual:
            return D3D11_COMPARISON_GREATER_EQUAL;
        case ir::ComparisonFunc::kLess:
            return D3D11_COMPARISON_LESS;
        case ir::ComparisonFunc::kLessEqual:
            return D3D11_COMPARISON_LESS_EQUAL;
        case ir::ComparisonFunc::kNever:
            return D3D11_COMPARISON_NEVER;
        case ir::ComparisonFunc::kNotEqual:
            return D3D11_COMPARISON_NOT_EQUAL;
        default:
            {
                assert(false && "Failed to colwert D3D11_COMPARISON_FUNC");
                return D3D11_COMPARISON_ALWAYS;
            }
        }
    };

    D3D11_STENCIL_OP colwertStencilOpD3D11(const ir::StencilOp & irStencilOp)
    {
        switch (irStencilOp)
        {
        case ir::StencilOp::kDecr:
            return D3D11_STENCIL_OP_DECR;
        case ir::StencilOp::kDecrSat:
            return D3D11_STENCIL_OP_DECR_SAT;
        case ir::StencilOp::kIncr:
            return D3D11_STENCIL_OP_INCR;
        case ir::StencilOp::kIncrSat:
            return D3D11_STENCIL_OP_INCR_SAT;
        case ir::StencilOp::kIlwert:
            return D3D11_STENCIL_OP_ILWERT;
        case ir::StencilOp::kKeep:
            return D3D11_STENCIL_OP_KEEP;
        case ir::StencilOp::kReplace:
            return D3D11_STENCIL_OP_REPLACE;
        case ir::StencilOp::kZero:
            return D3D11_STENCIL_OP_ZERO;
        default:
            {
                assert(false && "Failed to colwert D3D11_STENCIL_OP");
                return D3D11_STENCIL_OP_KEEP;
            }
        }
    };

    D3D11_DEPTH_STENCILOP_DESC colwertDepthStencilOpD3D11(const ir::DepthStencilOp & irDepthStencilOp)
    {
        D3D11_DEPTH_STENCILOP_DESC d3dDepthStencilOp;

        d3dDepthStencilOp.StencilFailOp = colwertStencilOpD3D11(irDepthStencilOp.failOp);
        d3dDepthStencilOp.StencilDepthFailOp = colwertStencilOpD3D11(irDepthStencilOp.depthFailOp);
        d3dDepthStencilOp.StencilPassOp = colwertStencilOpD3D11(irDepthStencilOp.depthFailOp);
        d3dDepthStencilOp.StencilFunc = colwertComparisonFuncD3D11(irDepthStencilOp.func);

        return d3dDepthStencilOp;
    };

    D3D11_DEPTH_WRITE_MASK colwertDepthWriteMaskD3D11(const ir::DepthWriteMask & irDepthWriteMask)
    {
        switch (irDepthWriteMask)
        {
        case ir::DepthWriteMask::kAll:
            return D3D11_DEPTH_WRITE_MASK_ALL;
        case ir::DepthWriteMask::kZero:
            return D3D11_DEPTH_WRITE_MASK_ZERO;
        default:
            {
                assert(false && "Failed to colwert D3D11_DEPTH_WRITE_MASK");
                return D3D11_DEPTH_WRITE_MASK_ALL;
            }
        }
    };

    D3D11_DEPTH_STENCIL_DESC CmdProcColwertDepthStencilStateD3D11(const ir::DepthStencilState & irDepthStencilState)
    {
        D3D11_DEPTH_STENCIL_DESC d3dDepthStencilState;

        d3dDepthStencilState.FrontFace = colwertDepthStencilOpD3D11(irDepthStencilState.frontFace);
        d3dDepthStencilState.BackFace = colwertDepthStencilOpD3D11(irDepthStencilState.backFace);

        d3dDepthStencilState.DepthWriteMask = colwertDepthWriteMaskD3D11(irDepthStencilState.depthWriteMask);
        d3dDepthStencilState.DepthFunc = colwertComparisonFuncD3D11(irDepthStencilState.depthFunc);

        d3dDepthStencilState.StencilReadMask = (UINT8)irDepthStencilState.stencilReadMask;
        d3dDepthStencilState.StencilWriteMask = (UINT8)irDepthStencilState.stencilWriteMask;
        d3dDepthStencilState.DepthEnable = (BOOL)(irDepthStencilState.isDepthEnabled != 0);
        d3dDepthStencilState.StencilEnable = (BOOL)(irDepthStencilState.isStencilEnabled != 0);

        return d3dDepthStencilState;
    };



    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // AlphaBlend State
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    D3D11_BLEND colwertBlendCoefD3D11(const ir::BlendCoef & irBlendCoef)
    {
        switch (irBlendCoef)
        {
        case ir::BlendCoef::kZero:
            return D3D11_BLEND_ZERO;
        case ir::BlendCoef::kOne:
            return D3D11_BLEND_ONE;
        case ir::BlendCoef::kSrcColor:
            return D3D11_BLEND_SRC_COLOR;
        case ir::BlendCoef::kIlwSrcColor:
            return D3D11_BLEND_ILW_SRC_COLOR;
        case ir::BlendCoef::kSrcAlpha:
            return D3D11_BLEND_SRC_ALPHA;
        case ir::BlendCoef::kIlwSrcAlpha:
            return D3D11_BLEND_ILW_SRC_ALPHA;
        case ir::BlendCoef::kDstAlpha:
            return D3D11_BLEND_DEST_ALPHA;
        case ir::BlendCoef::kIlwDstAlpha:
            return D3D11_BLEND_ILW_DEST_ALPHA;
        case ir::BlendCoef::kDstColor:
            return D3D11_BLEND_DEST_COLOR;
        case ir::BlendCoef::kIlwDstColor:
            return D3D11_BLEND_ILW_DEST_COLOR;
        case ir::BlendCoef::kSrcAlphaSat:
            return D3D11_BLEND_SRC_ALPHA_SAT;
        case ir::BlendCoef::kBlendFactor:
            return D3D11_BLEND_BLEND_FACTOR;
        case ir::BlendCoef::kIlwBlendFactor:
            return D3D11_BLEND_ILW_BLEND_FACTOR;
        case ir::BlendCoef::kSrc1Color:
            return D3D11_BLEND_SRC1_COLOR;
        case ir::BlendCoef::kIlwSrc1Color:
            return D3D11_BLEND_ILW_SRC1_COLOR;
        case ir::BlendCoef::kSrc1Alpha:
            return D3D11_BLEND_SRC1_ALPHA;
        case ir::BlendCoef::kIlwSrc1Alpha:
            return D3D11_BLEND_ILW_SRC1_ALPHA;
        default:
            {
                assert(false && "Failed to colwert D3D11_BLEND");
                return D3D11_BLEND_ONE;
            }
        }
    };

    D3D11_BLEND_OP colwertBlendOpD3D11(const ir::BlendOp & irBlendOp)
    {
        switch (irBlendOp)
        {
        case ir::BlendOp::kAdd:
            return D3D11_BLEND_OP_ADD;
        case ir::BlendOp::kSub:
            return D3D11_BLEND_OP_SUBTRACT;
        case ir::BlendOp::kRevSub:
            return D3D11_BLEND_OP_REV_SUBTRACT;
        case ir::BlendOp::kMin:
            return D3D11_BLEND_OP_MIN;
        case ir::BlendOp::kMax:
            return D3D11_BLEND_OP_MAX;
        default:
            {
                assert(false && "Failed to colwert D3D11_BLEND_OP");
                return D3D11_BLEND_OP_ADD;
            }
        }
    };

    D3D11_COLOR_WRITE_ENABLE colwertColorWriteEnableBitsD3D11(const ir::ColorWriteEnableBits & irColorWriteEnable)
    {
        switch (irColorWriteEnable)
        {
        case ir::ColorWriteEnableBits::kRed:
            return D3D11_COLOR_WRITE_ENABLE_RED;
        case ir::ColorWriteEnableBits::kGreen:
            return D3D11_COLOR_WRITE_ENABLE_GREEN;
        case ir::ColorWriteEnableBits::kBlue:
            return D3D11_COLOR_WRITE_ENABLE_BLUE;
        case ir::ColorWriteEnableBits::kAlpha:
            return D3D11_COLOR_WRITE_ENABLE_ALPHA;
        case ir::ColorWriteEnableBits::kAll:
            return D3D11_COLOR_WRITE_ENABLE_ALL;
        default:
            {
                assert(false && "Failed to colwert D3D11_COLOR_WRITE_ENABLE");
                return D3D11_COLOR_WRITE_ENABLE_ALL;
            }
        }
    };

    D3D11_RENDER_TARGET_BLEND_DESC colwertAlphaBlendRenderTargetStateD3D11(const ir::AlphaBlendRenderTargetState & irAlphaBlendRenderTargetState)
    {
        D3D11_RENDER_TARGET_BLEND_DESC d3dAlphaBlendRenderTargetState;

        d3dAlphaBlendRenderTargetState.SrcBlend = colwertBlendCoefD3D11(irAlphaBlendRenderTargetState.src);
        d3dAlphaBlendRenderTargetState.DestBlend = colwertBlendCoefD3D11(irAlphaBlendRenderTargetState.dst);
        d3dAlphaBlendRenderTargetState.BlendOp = colwertBlendOpD3D11(irAlphaBlendRenderTargetState.op);

        d3dAlphaBlendRenderTargetState.SrcBlendAlpha = colwertBlendCoefD3D11(irAlphaBlendRenderTargetState.srcAlpha);
        d3dAlphaBlendRenderTargetState.DestBlendAlpha = colwertBlendCoefD3D11(irAlphaBlendRenderTargetState.dstAlpha);
        d3dAlphaBlendRenderTargetState.BlendOpAlpha = colwertBlendOpD3D11(irAlphaBlendRenderTargetState.opAlpha);

        d3dAlphaBlendRenderTargetState.RenderTargetWriteMask = colwertColorWriteEnableBitsD3D11(irAlphaBlendRenderTargetState.renderTargetWriteMask);
        d3dAlphaBlendRenderTargetState.BlendEnable = (BOOL)(irAlphaBlendRenderTargetState.isEnabled != 0);

        return d3dAlphaBlendRenderTargetState;
    };

    D3D11_BLEND_DESC CmdProcColwertAlphaBlendStateD3D11(const ir::AlphaBlendState & irAlphaBlendState)
    {
        D3D11_BLEND_DESC d3dAlphaBlendState;

        for (int idx = 0; idx < 8; ++idx)
        {
            d3dAlphaBlendState.RenderTarget[idx] = colwertAlphaBlendRenderTargetStateD3D11(irAlphaBlendState.renderTargetState[idx]);
        }

        d3dAlphaBlendState.AlphaToCoverageEnable = irAlphaBlendState.alphaToCoverageEnable;
        d3dAlphaBlendState.IndependentBlendEnable = irAlphaBlendState.independentBlendEnable;

        return d3dAlphaBlendState;
    };
}
