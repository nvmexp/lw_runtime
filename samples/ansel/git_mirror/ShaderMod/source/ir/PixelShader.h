#pragma once

#include "ShaderHelpers.h"

namespace shadermod
{
namespace ir
{

	class PixelShader
	{
	public:

		// TODO: add destructor that explicitely destroys created IncludeFileDescs
		PixelShader::PixelShader(const wchar_t * fileName, const char * entryPoint);
		PixelShader::~PixelShader();

		wchar_t	m_fileName[IR_FILENAME_MAX];
		char	m_entryPoint[IR_FILENAME_MAX];

		ID3D11PixelShader *			m_D3DPixelShader = nullptr;
		ID3D11ShaderReflection *	m_D3DReflection = nullptr;

		shaderhelpers::IncludeFileDesc m_rootFile;

		void setTraverseStackPtr(std::vector<int> * pixelShadersTraverseStack);

	protected:

		std::vector<int> * m_pixelShadersTraverseStack = nullptr;
	};

}
}
