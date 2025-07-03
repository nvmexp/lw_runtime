#pragma once

#include <vector>
#include <string>
#include <set>

#include "ir/Defines.h"
#include "ir/SpecializedPool.h"
#include "D3DCompilerHandler.h"

namespace shadermod
{
namespace ir
{
namespace shaderhelpers
{
    class IncludeFileDesc
    {
    public:

        void computePath();

        IncludeFileDesc * m_parent = nullptr;
        std::vector<IncludeFileDesc *> m_children;

        wchar_t m_filenameFull[IR_FILENAME_MAX];  // Path+filename
        // TODO: we can store only length here, for memory conservation
        wchar_t m_filepath[IR_FILENAME_MAX];    // Path ONLY

    protected:


    };

    class IncludeHandler : public ID3DInclude
    {
    public:
        IncludeHandler(const wchar_t * basePath, Pool<IncludeFileDesc> * inclFileAllocationPool);

        void setRootIncludeFile(IncludeFileDesc * rootIncludeFile);

        void setSystemBasePath(const wchar_t * sysBasePath);

        HRESULT __stdcall Open(D3D_INCLUDE_TYPE includeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID *ppData, UINT *pBytes);
        HRESULT __stdcall Close(LPCVOID pData);

        const std::set<std::wstring>& getUniqueIncludeFiles() const
        {
            return m_uniqueIncludeFiles;
        }

    private:
        std::set<std::wstring> m_uniqueIncludeFiles;

        wchar_t m_basePath[IR_FILENAME_MAX];
        wchar_t m_sysBasePath[IR_FILENAME_MAX];

        IncludeFileDesc * m_pRootIncludeFile = nullptr;
        IncludeFileDesc * m_pLwrrentIncludeFile = nullptr;
        Pool<IncludeFileDesc> * m_pIncludeFilePool = nullptr;

    };

    HRESULT compileShaderFromFile(const WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut, 
                                  ID3DInclude* pInclude, D3DCompilerHandler& d3dCompiler, ID3D11ShaderReflection** ppReflection = nullptr,
                                  std::string* errorString = nullptr);

    enum class ShaderType
    {
        kVertex = 0,
        kPixel = 1,

        kNUM_ENTRIES
    };

    HRESULT compileEffectShaderFromFileOrCache(ShaderType type, const wchar_t* shaderFullFileName, LPCSTR szEntryPoint,
        const wchar_t* effectsFolderPath, const wchar_t* tempFolderPath,
        ID3DBlob** ppBlobOut, IncludeHandler* pInclude, D3DCompilerHandler& d3dCompiler, ID3D11ShaderReflection** ppReflection = nullptr,
        std::string* errorString = nullptr);


}
}
}
