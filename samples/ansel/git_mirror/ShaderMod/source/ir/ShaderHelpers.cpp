#include <stdio.h>
#include <shlwapi.h>  // PathIsRelative
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <locale>

#include <d3d11_1.h>
#include <d3dcompiler.h>

#include "D3DCompilerHandler.h"
#include "Utils.h"

#include "ir/Defines.h"
#include "ir/SpecializedPool.h"
#include "ir/ShaderHelpers.h"
#include "ir/FileHelpers.h"
#include "Log.h"
#include "HardcodedFX.h"
#include "acef.h"

#include "darkroom/StringColwersion.h"

namespace shadermod
{
namespace ir
{
namespace shaderhelpers
{
    void IncludeFileDesc::computePath()
    {
        swprintf_s(m_filepath, IR_FILENAME_MAX, L"%s", m_filenameFull);
        for (ptrdiff_t i = wcslen(m_filepath) - 1; i >= 0; --i)
        {
            if (m_filepath[i] == L'\\' || m_filepath[i] == L'/')
            {
                m_filepath[i] = L'\0';
                break;
            }
        }
    }

    IncludeHandler::IncludeHandler(const wchar_t * basePath, Pool<IncludeFileDesc> * inclFileAllocationPool):
        m_pIncludeFilePool(inclFileAllocationPool)
    {
        m_sysBasePath[0] = L'\0';  // By default, no special treatment of system includes
        swprintf_s(m_basePath, IR_FILENAME_MAX, L"%s", basePath);
    }

    void IncludeHandler::setRootIncludeFile(IncludeFileDesc * rootIncludeFile)
    {
        m_pRootIncludeFile = rootIncludeFile;
        m_pLwrrentIncludeFile = rootIncludeFile;
    }

    void IncludeHandler::setSystemBasePath(const wchar_t * sysBasePath)
    {
        swprintf_s(m_sysBasePath, IR_FILENAME_MAX, L"%s", sysBasePath);
    }

    HRESULT __stdcall IncludeHandler::Open(D3D_INCLUDE_TYPE includeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID *ppData, UINT *pBytes)
    {
        wchar_t includedFilePath[IR_FILENAME_MAX];

        const unsigned int includeFileOpenMode = std::ios::in|std::ios::binary|std::ios::ate;
        std::ifstream includeFile;

        // First, try to search next to shader if it is "local" include
        if (includeType == D3D_INCLUDE_LOCAL)
        {
            // 1. Look in the same directory as the file that has the '#include ""' statement
            // 2. Look in the directories of the lwrrently opened include files, reverse order

            IncludeFileDesc * lwrIncludeFileDesc = m_pLwrrentIncludeFile;
            while (!includeFile.is_open() && lwrIncludeFileDesc != nullptr)
            {
                swprintf_s(includedFilePath, IR_FILENAME_MAX, L"%s\\%hs", lwrIncludeFileDesc->m_filepath, pFileName);
                includeFile.open(includedFilePath, includeFileOpenMode);
                lwrIncludeFileDesc = lwrIncludeFileDesc->m_parent;
            }
        }

        // Remaining logic applies for both "local" and <system> includes

        // 3. Check paths that are passed as Additional Include Directories
        // <NOT IMPLEMENTED> 4. Check paths stored in INCLUDE elw variable

        // TODO: more folders specified as additional include directories (possibly user-provided ptr+size)
        const int maxNumTries = 1;
        int tries = 0;
        while (!includeFile.is_open())
        {
            if (m_sysBasePath[0] != '\0')
            {
                swprintf_s(includedFilePath, IR_FILENAME_MAX, L"%s\\%hs", m_sysBasePath, pFileName);
            }
            else
            {
                swprintf_s(includedFilePath, IR_FILENAME_MAX, L"%s\\%hs", m_basePath, pFileName);
            }

            includeFile.open(includedFilePath, includeFileOpenMode);

            ++tries;
            if (tries == maxNumTries)
                break;
        }

        if (includeFile.is_open())
        {
            m_uniqueIncludeFiles.insert(std::wstring(includedFilePath));

            IncludeFileDesc * childFileDesc = m_pIncludeFilePool->getElement();
            new (childFileDesc) IncludeFileDesc();
            m_pLwrrentIncludeFile->m_children.push_back(childFileDesc);
            childFileDesc->m_parent = m_pLwrrentIncludeFile;

            // We'll be processing this include file next
            m_pLwrrentIncludeFile = childFileDesc;
            
            // Store path + filename (for further file ops)
            swprintf_s(childFileDesc->m_filenameFull, IR_FILENAME_MAX, L"%s", includedFilePath);

            // Store path only (for include handler colweniencce)
            childFileDesc->computePath();

            size_t fileSize = size_t(includeFile.tellg());

            // TODO: insert pool here
            char* includeFileData = new char[fileSize];

            includeFile.seekg(0, std::ios::beg);
            includeFile.read(includeFileData, fileSize);
            includeFile.close();

            *ppData = includeFileData;
            *pBytes = (UINT)fileSize;
        } else
        {
            // TODO[error]: Trigger error callback
            return E_FAIL;
        }

        return S_OK;
    }
    HRESULT __stdcall IncludeHandler::Close(LPCVOID pData)
    {
        m_pLwrrentIncludeFile = m_pLwrrentIncludeFile->m_parent;

        char* includeFileData = (char*)pData;
        delete [] includeFileData;
        return S_OK;
    }

    HRESULT compileShaderFromFile(const WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, 
                                  ID3DBlob** ppBlobOut, ID3DInclude* pInclude, D3DCompilerHandler& d3dCompiler, ID3D11ShaderReflection** ppReflection,
                                  std::string* errorString)
    {
        HRESULT hr = S_OK;

        DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;

        // Adding debug information to the shaders is disabled by default, as it affects how shader binary code looks,
        //  and thus violates hashing mechanics.
        // You can enable this if you need to, but then make sure to keep in mind the modding protection.
#if 0
#ifdef _DEBUG
        // Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
        // Setting this flag improves the shader debugging experience, but still allows 
        // the shaders to be optimized and to run exactly the way they will run in 
        // the release configuration of this program.
        dwShaderFlags |= D3DCOMPILE_DEBUG;

        // Disable optimizations to further improve shader debugging
        dwShaderFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
#endif

        PFND3DCOMPILEFROMFILEFUNC pfnD3DCompileFromFile = d3dCompiler.getD3DCompileFromFileFunc();

        ID3DBlob* pErrorBlob = nullptr;
        hr = pfnD3DCompileFromFile(szFileName, nullptr, pInclude, szEntryPoint, szShaderModel,
            dwShaderFlags, 0, ppBlobOut, &pErrorBlob);
        
        if (FAILED(hr))
        {
            // TODO[error]: add various error messages here
            // TODO: allow logic here that processes line positions for user hookup
            
            if (errorString)
            {
                std::wstring wsFilename(szFileName);
                std::string strFilename = darkroom::getUtf8FromWstr(szFileName);
                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "File " << strFilename << std::endl << "Entry point: " << szEntryPoint << std::endl << "Error: ";
            
                if (hr == D3D11_ERROR_FILE_NOT_FOUND || hr == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND))
                {
                    strstream << "File not found!";
                }
                else if (pErrorBlob)
                {
                    strstream << reinterpret_cast<const char*>(pErrorBlob->GetBufferPointer());
                }
                else
                {
                }

                *errorString = buf.str();
            }
            
            if (pErrorBlob) 
                pErrorBlob->Release();

            return hr;
        }
        else
        {
            if (ppReflection)
            {
                PFND3DREFLECTFUNC pfnD3DReflect = d3dCompiler.getD3DReflectFunc();

                hr = pfnD3DReflect((*ppBlobOut)->GetBufferPointer(), (*ppBlobOut)->GetBufferSize(), IID_ID3D11ShaderReflection, (void **)ppReflection);
            
                if (FAILED(hr))
                {
                    if (errorString)
                    {
                        std::wstring wsFilename(szFileName);
                        std::string strFilename(darkroom::getUtf8FromWstr(szFileName));

                        std::stringbuf buf;
                        std::ostream strstream(&buf);
                        strstream << "File " << strFilename << std::endl << "Entry point: " << szEntryPoint << std::endl << "Reflection error: failed to reflect shader";

                        *errorString = buf.str();
                    }
                    
                    if (pErrorBlob) pErrorBlob->Release();
                    if (*ppBlobOut) (*ppBlobOut)->Release();
                    
                    return hr;
                }
            }
        }

        if (pErrorBlob) pErrorBlob->Release();

        return S_OK;
    }

    const uint64_t magicWordAndVersion = compilerMagicWordAndVersion;

    HRESULT readCompiledShaderFromFile(const wchar_t* szFileName, const wchar_t* effectsFolderPath, ID3DBlob** ppBlobOut, D3DCompilerHandler& d3dCompiler, ID3D11ShaderReflection** ppReflection,
        std::string* errorString)
    {
        HRESULT hr = S_OK;

        filehelpers::MappedFile cso(szFileName);

        const size_t sizeOfHeader = 2 * sizeof(uint64_t);

        if (!cso.isValid() || cso.end() - cso.begin() < sizeOfHeader)
            return S_FALSE;
    
        const unsigned char* dataPtr = cso.begin();
        
        LOG_INFO("File compiled with Magic Word: %llx\t\"%S\"", reinterpret_cast<const uint64_t*>(dataPtr)[0], szFileName);
        if (reinterpret_cast<const uint64_t*>(dataPtr)[0] != magicWordAndVersion)
        {
            LOG_INFO("            Needed Magic Word: %llx\tMagic Words do not match! File was compiled with a different compiler version. Need to recompile.", magicWordAndVersion);
            return S_FALSE;
        }

        dataPtr += sizeof(uint64_t);
    
        uint64_t numDependencies = reinterpret_cast<const uint64_t*>(dataPtr)[0];
        dataPtr += sizeof(uint64_t);

        if (cso.end() - dataPtr < (ptrdiff_t) ((sizeof(uint32_t) + sizeof(uint64_t)) * numDependencies))
            return S_FALSE;
        
        std::vector<uint64_t> fileTimes;
        fileTimes.resize((size_t) numDependencies);

        for (unsigned int i = 0; i < numDependencies; ++i)
        {
            fileTimes[i] = reinterpret_cast<const uint64_t*>(dataPtr)[i];
        }

        dataPtr += sizeof(uint64_t) * numDependencies;
        
        std::vector<uint32_t> stringSizes;
        stringSizes.resize((size_t) numDependencies);
        
        unsigned int aclwm = 0;

        for (unsigned int i = 0; i < numDependencies; ++i)
        {
            stringSizes[i] = reinterpret_cast<const uint32_t*>(dataPtr)[i];
            aclwm += stringSizes[i];
        }

        dataPtr += sizeof(uint32_t) * numDependencies;

        if (cso.end() - dataPtr < (ptrdiff_t) (sizeof(wchar_t) * aclwm))
            return S_FALSE;

        for (unsigned int i = 0; i < numDependencies; ++i)
        {
            std::wstring path;

            uint32_t strsz = stringSizes[i];
            path.resize(strsz);

            for (unsigned int j = 0; j < strsz; ++j)
                path[j] = reinterpret_cast<const wchar_t*>(dataPtr)[j];

            dataPtr += sizeof(wchar_t) * strsz;

            std::wstring effectNameNoExt = path.substr(0, path.find_last_of(L'.'));
            if (IsEffectHardcoded(effectNameNoExt))
            {
                continue;
            }
                    
            std::wstring fullPath = std::wstring(effectsFolderPath) + path;
                    
            uint64_t filetime = 0;

            if (!filehelpers::getFileTime(fullPath.c_str(), filetime))
                return S_FALSE;

            if (filetime != fileTimes[i])
                return S_FALSE;
        }

        PFND3DCREATEBLOBFUNC pfnD3DCreateBlob = d3dCompiler.getD3DCreateBlobFunc();

        hr = pfnD3DCreateBlob(cso.end() - dataPtr, ppBlobOut);

        if (FAILED(hr))
        {
            if (errorString)
            {
                std::wstring wsFilename(szFileName);
                std::string strFilename = darkroom::getUtf8FromWstr(szFileName);
                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "File " << strFilename << " Error: ";
                strstream << "Creating D3D Blob for the compiled shader failed!";

                *errorString = buf.str();
            }

            return hr;
        }

        memcpy((*ppBlobOut)->GetBufferPointer(), dataPtr, cso.end() - dataPtr);

        if (ppReflection)
        {
            PFND3DREFLECTFUNC pfnD3DReflect = d3dCompiler.getD3DReflectFunc();

            hr = pfnD3DReflect((*ppBlobOut)->GetBufferPointer(), (*ppBlobOut)->GetBufferSize(), IID_ID3D11ShaderReflection, (void **)ppReflection);

            if (FAILED(hr))
            {
                if (errorString)
                {
                    std::wstring wsFilename(szFileName);
                    std::string strFilename(darkroom::getUtf8FromWstr(szFileName));

                    std::stringbuf buf;
                    std::ostream strstream(&buf);
                    strstream << "File " << strFilename << " Reflection error: failed to reflect shader";

                    *errorString = buf.str();
                }

                return hr;
            }
        }
        
        return S_OK;
    }
    
    ptrdiff_t getShaderRelativeFileNameOffs(const wchar_t* shaderFullFileName,
                                                const wchar_t* effectsFolderPath, 
                                                const wchar_t* tempFolderPath,
                                                ptrdiff_t shaderFullFileNameLen,
                                                ptrdiff_t effectsFolderPathLen)
    {
        const wchar_t* shaderRelativeFileName = nullptr;

        if (effectsFolderPathLen < shaderFullFileNameLen)
        {
            const wchar_t *tempStr = effectsFolderPath;
            shaderRelativeFileName = shaderFullFileName;

            while (effectsFolderPathLen > 0 && *tempStr == *shaderRelativeFileName)
                ++tempStr, ++shaderRelativeFileName, --effectsFolderPathLen;

            if (effectsFolderPathLen != 0)
                shaderRelativeFileName = nullptr;
        }

        if (!shaderRelativeFileName)
            return -1;

        return shaderRelativeFileName - shaderFullFileName;
    }

    HRESULT writeCompiledShaderToFile(const wchar_t* shaderFullFileName, LPCSTR szEntryPoint, 
        const wchar_t* effectsFolderPath, const wchar_t* tempFolderPath,
        const std::wstring & compiledShaderPathOnly, const std::wstring & compiledShaderFileName,
        const wchar_t* shaderRelativeFileName, ptrdiff_t relPathOffs,
        ID3DBlob** ppBlobOut, IncludeHandler* pInclude,
        std::string* errorString)
    {
        ptrdiff_t shaderFullFileNameLen = wcslen(shaderFullFileName);
        ptrdiff_t effectsFolderPathLen = wcslen(effectsFolderPath);

        if (!filehelpers::createDirectoryRelwrsively(compiledShaderPathOnly.c_str()))
        {
            if (errorString)
            {
                std::string strFilename = darkroom::getUtf8FromWstr(shaderFullFileName);
                std::string strEffectFolder = darkroom::getUtf8FromWstr(effectsFolderPath);
                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "File " << strFilename << std::endl << "Entry point: " << szEntryPoint << std::endl << "Error: ";
                strstream << "Can't create cached shaders path: " << darkroom::getUtf8FromWstr(compiledShaderPathOnly);

                *errorString = buf.str();
            }

            return E_FAIL;
        }

        std::ofstream out(compiledShaderFileName, std::ofstream::binary);

        if (!out)
        {
            if (errorString)
            {
                std::string strFilename = darkroom::getUtf8FromWstr(shaderFullFileName);
                std::string strEffectFolder = darkroom::getUtf8FromWstr(effectsFolderPath);
                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "File " << strFilename << std::endl << "Entry point: " << szEntryPoint << std::endl << "Error: ";
                strstream << "Can't create cached shader file: " << darkroom::getUtf8FromWstr(compiledShaderFileName);

                *errorString = buf.str();
            }

            return E_FAIL;
        }

        auto& includeFiles = pInclude->getUniqueIncludeFiles();
        uint64_t numDependencies = 1 + includeFiles.size();
        std::vector<ptrdiff_t> relativeIncludeFilePathOffs;
        relativeIncludeFilePathOffs.reserve(includeFiles.size());

        std::vector<uint64_t> fileTimes;
        fileTimes.reserve((size_t) numDependencies);

        std::vector<uint32_t> stringSizes;
        stringSizes.reserve((size_t) numDependencies);

        uint64_t filetime = 0;

        if (!filehelpers::getFileTime(shaderFullFileName, filetime))
        {
            if (errorString)
            {
                std::string strFilename = darkroom::getUtf8FromWstr(shaderFullFileName);
                std::string strEffectFolder = darkroom::getUtf8FromWstr(effectsFolderPath);
                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "File " << strFilename << std::endl << "Entry point: " << szEntryPoint << std::endl << "Error: ";
                strstream << "Can't open shader source!";

                *errorString = buf.str();
            }

            return E_FAIL;
        }

        fileTimes.push_back(filetime);
        stringSizes.push_back((unsigned int) (shaderFullFileNameLen - relPathOffs));

        for (auto& i : includeFiles)
        {
            if (!filehelpers::getFileTime(i.c_str(), filetime))
            {
                if (errorString)
                {
                    std::string strFilename = darkroom::getUtf8FromWstr(i.c_str());
                    std::string strEffectFolder = darkroom::getUtf8FromWstr(effectsFolderPath);
                    std::stringbuf buf;
                    std::ostream strstream(&buf);
                    strstream << "File " << strFilename << std::endl << "Entry point: " << szEntryPoint << std::endl << "Error: ";
                    strstream << "Can't open shader header!";

                    *errorString = buf.str();
                }

                return E_FAIL;
            }

            fileTimes.push_back(filetime);

            ptrdiff_t relativePathOffs = getShaderRelativeFileNameOffs(i.c_str(),
                                            effectsFolderPath,
                                            tempFolderPath,
                                            i.length(),
                                            effectsFolderPathLen);

            if (relativePathOffs < 0)
            {
                if (errorString)
                {
                    std::string strFilename = darkroom::getUtf8FromWstr(i.c_str());
                    std::string strEffectFolder = darkroom::getUtf8FromWstr(effectsFolderPath);
                    std::stringbuf buf;
                    std::ostream strstream(&buf);
                    strstream << "File " << strFilename << " Error: ";
                    strstream << "File path not relative to the effects folder: " << strEffectFolder;

                    *errorString = buf.str();
                }

                return E_FAIL;
            }

            relativeIncludeFilePathOffs.push_back(relativePathOffs);
            const wchar_t* includeRelativeFileName = i.c_str() + relativePathOffs;
            stringSizes.push_back((unsigned int) (i.length() - relativePathOffs));
        }

        assert(fileTimes.size() == stringSizes.size());
        assert(fileTimes.size() == numDependencies);

        out.write(reinterpret_cast<const char*>(&magicWordAndVersion), sizeof(uint64_t));
        out.write(reinterpret_cast<const char*>(&numDependencies), sizeof(uint64_t));
        out.write(reinterpret_cast<const char*>(fileTimes.data()), numDependencies * sizeof(uint64_t));
        out.write(reinterpret_cast<const char*>(stringSizes.data()), numDependencies * sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(shaderRelativeFileName),
                        (shaderFullFileNameLen - relPathOffs) * sizeof(wchar_t));

        ptrdiff_t* offsCtr = relativeIncludeFilePathOffs.data();

        for (auto& i : includeFiles)
        {
            out.write(reinterpret_cast<const char*>(i.c_str() + *offsCtr), (i.length() - *offsCtr) * sizeof(wchar_t));
            offsCtr += 1;
        }

        out.write(reinterpret_cast<char*>((*ppBlobOut)->GetBufferPointer()), (*ppBlobOut)->GetBufferSize());

        return S_OK;
    }

    HRESULT compileEffectShaderFromFileOrCache(ShaderType type, const wchar_t* shaderFullFileName, LPCSTR szEntryPoint, 
        const wchar_t* effectsFolderPath, const wchar_t* tempFolderPath,
        ID3DBlob** ppBlobOut, IncludeHandler* pInclude, D3DCompilerHandler& d3dCompiler, ID3D11ShaderReflection** ppReflection,
        std::string* errorString)
    {
        const char* effectsShaderVersionString = (type == ShaderType::kVertex) ? "vs_4_0" : "ps_4_0"; //this is ilwariant

        //check if the cached version of the file is present
        ptrdiff_t shaderFullFileNameLen = wcslen(shaderFullFileName);
        ptrdiff_t effectsFolderPathLen = wcslen(effectsFolderPath);

        const ptrdiff_t relPathOffs = getShaderRelativeFileNameOffs(shaderFullFileName,
                                                                        effectsFolderPath,
                                                                        tempFolderPath,
                                                                        shaderFullFileNameLen,
                                                                        effectsFolderPathLen);
                
        if (relPathOffs < 0)
        {
            if (errorString)
            {
                std::string strFilename = darkroom::getUtf8FromWstr(shaderFullFileName);
                std::string strEffectFolder = darkroom::getUtf8FromWstr(effectsFolderPath);
                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "File " << strFilename << std::endl << "Entry point: " << szEntryPoint << std::endl << "Error: ";
                strstream << "File path not relative to the effects folder: " << strEffectFolder;

                *errorString = buf.str();
            }

            return E_ILWALIDARG;
        }

        const wchar_t* shaderRelativeFileName = shaderFullFileName + relPathOffs;

        std::wstring wsEntryPoint = darkroom::getWstrFromUtf8(szEntryPoint);

        std::wstring compiledShaderPathOnly = std::wstring(tempFolderPath) + std::wstring(L"_pscache\\") + std::wstring(shaderRelativeFileName) +
            std::wstring(L"\\");

        std::wstring compiledShaderFileName = compiledShaderPathOnly + wsEntryPoint + std::wstring(L".cso");

        std::wstring shaderRelativeFileNameNoExt(shaderRelativeFileName);
        shaderRelativeFileNameNoExt = shaderRelativeFileNameNoExt.substr(0, shaderRelativeFileNameNoExt.find_last_of(L'.'));
        
        if (IsEffectHardcoded(shaderRelativeFileNameNoExt))
        {
            WriteHardcodedBinaryFile(shaderRelativeFileNameNoExt, compiledShaderFileName);
            if (!shadermod::ir::filehelpers::validateFileExists(compiledShaderFileName))
            {
                return E_FAIL;
            }
        }

        HRESULT hr = E_FAIL;

        if (filehelpers::directoryExists(compiledShaderPathOnly.c_str()))
            hr = readCompiledShaderFromFile(compiledShaderFileName.c_str(), effectsFolderPath, ppBlobOut, d3dCompiler, ppReflection, nullptr);

        if (SUCCEEDED(hr) && hr != S_FALSE)
            return hr;

        //try to compile from source
        hr = compileShaderFromFile(shaderFullFileName, szEntryPoint, effectsShaderVersionString, ppBlobOut, pInclude, d3dCompiler,
            ppReflection, errorString);

        if (FAILED(hr))
            return hr;

        const bool attemptToCacheShaders = true;
        if (attemptToCacheShaders)
        {
            hr = writeCompiledShaderToFile(shaderFullFileName, szEntryPoint, 
                        effectsFolderPath, tempFolderPath,
                        compiledShaderPathOnly, compiledShaderFileName,
                        shaderRelativeFileName, relPathOffs,
                        ppBlobOut, pInclude,
                        errorString);

            if (FAILED(hr))
            {
                std::string strFilename = darkroom::getUtf8FromWstr(shaderFullFileName);
                std::stringbuf buf;
                std::ostream strstream(&buf);
                strstream << "File " << strFilename << std::endl << "Entry point: " << szEntryPoint << std::endl << "Error: ";
                strstream << "Couldn't create shader cache";

                LOG_WARN(buf.str().c_str());

                // Not a fatal error
                hr = S_OK;
            }
        }

        return S_OK;
    }
}
}
}