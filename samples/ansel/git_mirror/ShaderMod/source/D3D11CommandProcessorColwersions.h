#pragma once

#include <stdint.h>
#include "ir/PipelineStates.h"

namespace shadermod
{

    D3D11_RASTERIZER_DESC CmdProcColwertRasterizerStateD3D11(const ir::RasterizerState & irRasterState);
    D3D11_DEPTH_STENCIL_DESC CmdProcColwertDepthStencilStateD3D11(const ir::DepthStencilState & irDepthStencilState);
    D3D11_BLEND_DESC CmdProcColwertAlphaBlendStateD3D11(const ir::AlphaBlendState & irAlphaBlendState);

}