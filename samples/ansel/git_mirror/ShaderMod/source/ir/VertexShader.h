#pragma once

namespace shadermod
{
namespace ir
{

    class VertexShader
    {
    public:

        // TODO: add destructor that explicitely destroys created IncludeFileDescs
        VertexShader::VertexShader(const wchar_t * fileName, const char * entryPoint);
        VertexShader::~VertexShader();

        wchar_t  m_fileName[IR_FILENAME_MAX];
        char  m_entryPoint[IR_FILENAME_MAX];

        ID3D11VertexShader *    m_D3DVertexShader = nullptr;
        ID3D11ShaderReflection *  m_D3DReflection = nullptr;

        shaderhelpers::IncludeFileDesc m_rootFile;

        void setTraverseStackPtr(std::vector<int> * vertexShadersTraverseStack);

    protected:

        std::vector<int> * m_vertexShadersTraverseStack = nullptr;
    };

}
}
