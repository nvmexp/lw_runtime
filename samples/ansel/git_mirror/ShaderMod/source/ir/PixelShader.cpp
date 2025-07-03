#include <stdio.h>
#include <vector>

#include <d3d11_1.h>
#include <D3D11Shader.h>

#include "ir/Defines.h"
#include "ir/SpecializedPool.h"
#include "ir/ShaderHelpers.h"
#include "ir/PixelShader.h"

namespace shadermod
{
namespace ir
{

    PixelShader::PixelShader(const wchar_t * fileName, const char * entryPoint)
    {
        swprintf_s(m_fileName, IR_FILENAME_MAX, L"%s", fileName);
        sprintf_s(m_entryPoint, IR_FILENAME_MAX, "%s", entryPoint);
    }

    PixelShader::~PixelShader()
    {
        if (m_D3DReflection)
        {
            m_D3DReflection->Release();
            m_D3DReflection = nullptr;
        }

        shaderhelpers::IncludeFileDesc * pFileDesc = &m_rootFile;

        m_pixelShadersTraverseStack->clear();
        m_pixelShadersTraverseStack->reserve(5);
        m_pixelShadersTraverseStack->push_back(0);
        while (pFileDesc)
        {
            int lwrrentChild = m_pixelShadersTraverseStack->back();
            if (pFileDesc->m_children.size() > size_t(lwrrentChild))
            {
                pFileDesc = pFileDesc->m_children[lwrrentChild];
                m_pixelShadersTraverseStack->push_back(0);
            }
            else
            {
                shaderhelpers::IncludeFileDesc * pFileDescDestroy = pFileDesc;
                pFileDesc = pFileDesc->m_parent;

                pFileDescDestroy->~IncludeFileDesc();

                if (pFileDesc)
                {
                    m_pixelShadersTraverseStack->pop_back();
                    m_pixelShadersTraverseStack->back() += 1;
                }
            }
        }
    }

    void PixelShader::setTraverseStackPtr(std::vector<int> * pixelShadersTraverseStack)
    {
        m_pixelShadersTraverseStack = pixelShadersTraverseStack;
    }

}
}
