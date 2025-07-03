#include <stdio.h>
#include <vector>

#include <d3d11_1.h>
#include <D3D11Shader.h>

#include "ir/Defines.h"
#include "ir/SpecializedPool.h"
#include "ir/ShaderHelpers.h"
#include "ir/VertexShader.h"

namespace shadermod
{
namespace ir
{

    VertexShader::VertexShader(const wchar_t * fileName, const char * entryPoint)
    {
        swprintf_s(m_fileName, IR_FILENAME_MAX, L"%s", fileName);
        sprintf_s(m_entryPoint, IR_FILENAME_MAX, "%s", entryPoint);
    }

    VertexShader::~VertexShader()
    {
        if (m_D3DReflection)
        {
            m_D3DReflection->Release();
            m_D3DReflection = nullptr;
        }

        shaderhelpers::IncludeFileDesc * pFileDesc = &m_rootFile;

        m_vertexShadersTraverseStack->clear();
        m_vertexShadersTraverseStack->reserve(5);
        m_vertexShadersTraverseStack->push_back(0);
        while (pFileDesc)
        {
            int lwrrentChild = m_vertexShadersTraverseStack->back();
            if (pFileDesc->m_children.size() > size_t(lwrrentChild))
            {
                pFileDesc = pFileDesc->m_children[lwrrentChild];
                m_vertexShadersTraverseStack->push_back(0);
            }
            else
            {
                shaderhelpers::IncludeFileDesc * pFileDescDestroy = pFileDesc;
                pFileDesc = pFileDesc->m_parent;

                pFileDescDestroy->~IncludeFileDesc();

                if (pFileDesc)
                {
                    m_vertexShadersTraverseStack->pop_back();
                    m_vertexShadersTraverseStack->back() += 1;
                }
            }
        }
    }

    void VertexShader::setTraverseStackPtr(std::vector<int> * vertexShadersTraverseStack)
    {
        m_vertexShadersTraverseStack = vertexShadersTraverseStack;
    }

}
}
