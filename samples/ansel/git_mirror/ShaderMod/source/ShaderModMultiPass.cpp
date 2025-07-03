#include <algorithm>
#include <stdio.h>
#include <fstream>
#include <shlwapi.h>    // PathIsRelative
#include <string.h>
#include <unordered_map>
#include <iomanip>
#include <array>

#include <windows.h>
#include <d3d11_1.h>
#include <ir/ShaderHelpers.h>

#include <vector>
#include <list>
#include <algorithm>
#include <assert.h>

// Certain states (RenderTargets and ShaderResources) need to be reset after Draw in order to avoid potential rendering bugs
#define CLEANUP_STATES  1

// since we only draw full-screen triangles, we shouldn't have to clean up RT views,right? Scaled passes should simply yield worse precision..
#define FORCE_CLEAR_RT_VIEW 0

#include "Utils.h"
#include "sha2.hpp"

#include "D3DCompilerHandler.h"
#include "MultipassConfigParserError.h"
#include "ResourceManager.h"
#include "D3D11CommandProcessor.h"
#include "ir/UserConstantManager.h"
#include "ir/IRCPPHeaders.h"
#include "ir/BinaryColwersion.h"
#include "ir/FileHelpers.h"
#include "ShaderModMultiPass.h"
#include "drawTriangleVS.h"
#include "copyPS.h"
#include "darkroom/StringColwersion.h"
#include "Log.h"
#include "HardcodedFX.h"
#include "RenderBuffer.h"

#include "source/lwbins/sharpen_sm30_ldr.h"
#include "source/lwbins/sharpen_sm30_hdr.h"
#include "source/lwbins/sharpen_sm50_ldr.h"
#include "source/lwbins/sharpen_sm50_hdr.h"
#include "source/lwbins/sharpen_sm60_ldr.h"
#include "source/lwbins/sharpen_sm60_hdr.h"
#include "source/lwbins/sharpen_sm75_ldr.h"
#include "source/lwbins/sharpen_sm75_hdr.h"
#include "source/lwbins/sharpen_sm86_ldr.h"
#include "source/lwbins/sharpen_sm86_hdr.h"
#include "source/api/lwsdk_ngx_parameters.h"

#include "acef.h"

typedef unsigned int uint;

#define CHECK_RESULT(hr, msg) if (FAILED(hr)) return false;

bool isHdrFormatSupported(DXGI_FORMAT format)
{
    static const std::set<DXGI_FORMAT> s_SupportedHdrFormats = {
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32_FLOAT,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R11G11B10_FLOAT
    };
    return s_SupportedHdrFormats.find(format) != s_SupportedHdrFormats.cend();
}

void LogResourceDescription(std::string title, NGXLwbin_Resource_Description desc)
{
    LOG_DEBUG("Resource Description: %s", title.c_str());
    LOG_DEBUG("  (Width,Height): (%d,%d)", desc.Width, desc.Height);
    LOG_DEBUG("  Format: %s", DxgiFormat_cstr(desc.Format));
    LOG_DEBUG("  Mips: %d", desc.Mips);
    LOG_DEBUG("  GPUVirtualAddress: 0x%llx", desc.GPUVirtualAddress);
    LOG_DEBUG("  Properties:");
    if (desc.Properties == 0) { LOG_DEBUG("    NONE"); }
    if (desc.Properties & NGXLwbin_Resource_Property_Tex2D) { LOG_DEBUG("    NGXLwbin_Resource_Property_Tex2D"); }
    if (desc.Properties & NGXLwbin_Resource_Property_Buffer) { LOG_DEBUG("    NGXLwbin_Resource_Property_Buffer"); }
    if (desc.Properties & NGXLwbin_Resource_Property_UAV) { LOG_DEBUG("    NGXLwbin_Resource_Property_UAV"); }
}

static HRESULT GetLUIDFromD3D11Device(ID3D11Device *InDevice, LUID *OutLuid)
{
    IDXGIDevice * pDXGIDevice;
    HRESULT hr = InDevice->QueryInterface(__uuidof(IDXGIDevice), (void **)&pDXGIDevice);
    if (FAILED(hr))
    {
        LOG_ERROR("error: %s failed - unable to query DXGI Device", __func__);
        return hr;
    }

    IDXGIAdapter * pAdapter;
    hr = pDXGIDevice->GetAdapter(&pAdapter);
    if (FAILED(hr))
    {
        pDXGIDevice->Release();
        LOG_ERROR("error: %s failed - unable to get DXGI adapter", __func__);
        return hr;
    }

    DXGI_ADAPTER_DESC adapterDesc;
    hr = pAdapter->GetDesc(&adapterDesc);
    if (FAILED(hr))
    {
        pAdapter->Release();
        pDXGIDevice->Release();
        LOG_ERROR("error: %s failed - unable to get descriptor for DXGI Adapter", __func__);
        return hr;
    }

    memcpy(OutLuid, &adapterDesc.AdapterLuid, sizeof(LUID));

    pAdapter->Release();
    pDXGIDevice->Release();
    return S_OK;
}

static bool GetGPUArch(ID3D11Device *InDevice, LW_GPU_ARCHITECTURE_ID& gpuArchId)
{
    // #1 Enumerate the GPUs on the system returning physical GPU handles.
    LwPhysicalGpuHandle physicalGpuHandles[LWAPI_MAX_PHYSICAL_GPUS];
    LwU32 gpuCnt = 0;
    LwAPI_Status lwStatus = LwAPI_EnumPhysicalGPUs(physicalGpuHandles, &gpuCnt);
    if (lwStatus != LWAPI_OK)
    {
        LOG_ERROR("error: LwAPI_EnumPhysicalGPUs failed");
        return false;
    }

    // #2 Get LUID of the GPU adapter corresponding to the D3D device.
    LUID d3dDeviceLuid;
    HRESULT hr = GetLUIDFromD3D11Device(InDevice, &d3dDeviceLuid);
    if (FAILED(hr))
    {
        LOG_ERROR("error: GetLUIDFromD3D11Device failed");
        return false;
    }

    for (LwU32 gpuIndex = 0; gpuIndex < gpuCnt; gpuIndex++)
    {
        // #3 Get LUID of each enumerated GPU
        LUID adapterLUID = { 0 };
        lwStatus = LwAPI_GPU_GetAdapterIdFromPhysicalGpu(physicalGpuHandles[gpuIndex], &adapterLUID);
        if (lwStatus != LWAPI_OK)
        {
            LOG_ERROR("error: LwAPI_GPU_GetAdapterIdFromPhysicalGpu failed - unable to get Adapter LUID from LWAPI Physical GPU handle - potentially Windows remote-desktop scenario.");
            return false;
        }


        if (adapterLUID.HighPart == d3dDeviceLuid.HighPart && adapterLUID.LowPart == d3dDeviceLuid.LowPart)
        {
            // #4 If this LUID matches that of the D3D device, then return the corresponding GPU architecture.
            LW_GPU_ARCH_INFO Info = {};
            Info.version = LW_GPU_ARCH_INFO_VER_2;
            lwStatus = LwAPI_GPU_GetArchInfo(physicalGpuHandles[gpuIndex], &Info);
            if (lwStatus != LWAPI_OK)
            {
                LOG_ERROR("error: %s failed - unable to get GPU architecture info from physical GPU handle.");
                return false;
            }

            gpuArchId = Info.architecture_id;
            return true;
        }
    }

    LOG_ERROR("error: %s failed - Could not fetch GPU architecture corresponding to D3D device!", __func__);

    return false;
}

namespace shadermod
{
    class IncludeHandler : public ID3DInclude
    {
    public:
        IncludeHandler(const char * basePath)
        {
            sprintf_s(m_basePath, FILENAME_MAX*sizeof(char), "%s", basePath);
        }

        HRESULT __stdcall Open(D3D_INCLUDE_TYPE includeType, LPCSTR pFileName, LPCVOID pParentData,
            LPCVOID *ppData, UINT *pBytes)
        {
            // We do no allow absolute include paths
            if (!PathIsRelativeA(pFileName))
            {
                // TODO(error): Trigger error callback
                return E_FAIL;
            }
            // We do not allow '..\' in the include paths
            if (strstr(pFileName, "..\\") != 0)
            {
                // TODO(error): Trigger error callback
                return E_FAIL;
            }

            char includedFilePath[FILENAME_MAX];
            switch (includeType)
            {
                case D3D_INCLUDE_LOCAL:
                {
                    sprintf_s(includedFilePath, FILENAME_MAX*sizeof(char), "%s\\%s", m_basePath, pFileName);
                    break;
                }
                case D3D_INCLUDE_SYSTEM:
                {
                    // Probably treat sys includes differently
                    sprintf_s(includedFilePath, FILENAME_MAX*sizeof(char), "%s\\%s", m_basePath, pFileName);
                    break;
                }
                default:
                {
                    // TODO(error): Trigger error callback
                    return E_FAIL;
                }
            };

            std::ifstream includeFile(includedFilePath, std::ios::in | std::ios::binary | std::ios::ate);

            if (includeFile.is_open())
            {
                size_t fileSize = size_t(includeFile.tellg());

                // TODO: insert pool here
                char* includeFileData = new char[fileSize];

                includeFile.seekg(0, std::ios::beg);
                includeFile.read(includeFileData, fileSize);
                includeFile.close();

                *ppData = includeFileData;
                *pBytes = (UINT)fileSize;
            }
            else
            {
                return E_FAIL;
            }

            return S_OK;
        }
        HRESULT __stdcall Close(LPCVOID pData)
        {
            char* includeFileData = (char*)pData;
            delete[] includeFileData;
            return S_OK;
        }

    private:
        char m_basePath[FILENAME_MAX];
    };

    // TODO: unify the code with JobProcessing
    std::wstring generateCmdLine(const std::wstring& exelwtable, const std::vector<std::wstring>& args)
    {
        std::wostringstream argsJoined;
        std::copy(args.begin(), args.end(), std::ostream_iterator<std::wstring, wchar_t>(argsJoined, L" "));
        return exelwtable + L" " + argsJoined.str();
    }
    bool exelwteProcess(const std::wstring& exelwtable, const std::vector<std::wstring>& args, const std::wstring& workDir, HANDLE& process)
    {
        STARTUPINFO si;
        PROCESS_INFORMATION pi;

        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        ZeroMemory(&pi, sizeof(pi));

        std::wstring cmdLineString = generateCmdLine(exelwtable, args);
        wchar_t* cmdLine = const_cast<wchar_t*>(cmdLineString.c_str()); // ouch

        if (CreateProcess(NULL, cmdLine, NULL, NULL, FALSE, CREATE_UNICODE_ELWIRONMENT | NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW, NULL, workDir.c_str(), &si, &pi) == FALSE)
        {
            DWORD err = GetLastError();
            return false;
        }

        process = pi.hProcess;

        CloseHandle(pi.hThread);

        return true;
    }

    void MultiPassEffect::initializeEffect(
        const wchar_t * fxToolFilepath, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,
        const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
        const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
        const ir::Effect::InputData & colorBaseInput,
        const std::set<Hash::Effects> * pExpectedHashSet,
        bool compareHashes
        )
    {
        m_fxFilepath = m_rootDir + m_fxFilename;

        const bool acefTesting = true;
        bool acefSaving = true;

        std::wstring effectFullpath(rootDir);
        effectFullpath += fxFilename;

        LOG_VERBOSE("Effect loaded: %s, path: %s", darkroom::getUtf8FromWstr(fxFilename).c_str(), darkroom::getUtf8FromWstr(effectFullpath).c_str());

        size_t effectNameLen = wcslen(fxFilename);
        std::wstring effectName(fxFilename);
        size_t effectFolderNameLen = 0;
        for (int idx = (int)effectName.length() - 1; idx >= 0; --idx)
        {
            if (effectName[idx] == L'\\' || effectName[idx] == L'/')
                break;

            effectFolderNameLen = idx;
        }

        std::wstring binaryPathOnly;
        std::wstring binaryName;
        std::wstring binaryPathAndName;

        binaryPathOnly = std::wstring(tempsDir) + std::wstring(L"_binaries\\") + fxFilename;

        ir::filehelpers::createDirectoryRelwrsively((binaryPathOnly + L"\\").c_str());

        std::wstring effectNameWithExt(fxFilename + effectFolderNameLen, effectNameLen - effectFolderNameLen);
        std::wstring effectNameNoExt = effectNameWithExt.substr(0, effectNameWithExt.find_last_of(L'.'));
        binaryName = effectNameNoExt + L".acef";

        binaryPathAndName = binaryPathOnly + L"\\" + binaryName;

        m_fxToolFilepath = fxToolFilepath;
        // TODO : move this check under the needRecompilation condition check
        if (!ir::filehelpers::validateFileExists(m_fxToolFilepath))
        {
            std::string m_errorStrUTF8 = "Required tool not found: " + darkroom::getUtf8FromWstr(m_fxToolFilepath);
            throw MultipassConfigParserError(
                MultipassConfigParserErrorEnum::eInternalError,
                m_errorStrUTF8.c_str()
                );

            return;
        }

        acef::EffectStorage acefEffectStorageHeaderCheck;
        bool needRecompilation = false;

        acef::FileReadingStatus acefFileStatus = acef::loadHeaderData(binaryPathAndName.c_str(), &acefEffectStorageHeaderCheck);
        if (acefFileStatus != acef::FileReadingStatus::kOK)
        {
            needRecompilation = true;
        }
        else
        {
            auto matchFileTime = [](const wchar_t * filepath, uint64_t expectedTime, bool exactMatch) -> bool
            {
                uint64_t filetime;
                if (!shadermod::ir::filehelpers::getFileTime(filepath, filetime))
                {
                    return false;
                }

                if (exactMatch && (filetime != expectedTime))
                {
                    return false;
                }
                else if (!exactMatch && (filetime <= expectedTime))
                {
                    return false;
                }

                return true;
            };

            acef::Header & effectHeader = acefEffectStorageHeaderCheck.header;

            if (!matchFileTime(effectFullpath.c_str(), effectHeader.binStorage.timestamp, true))
            {
                needRecompilation = true;
            }
            else
            {
                for (uint32_t depIdx = 0, depNum = effectHeader.binStorage.dependenciesNum; depIdx < depNum; ++depIdx)
                {
                    uint64_t depTimestamp = effectHeader.fileTimestamps[depIdx];
                    std::string depFilenameUTF8 = std::string(effectHeader.filePathsUtf8 + effectHeader.filePathOffsets[depIdx], effectHeader.filePathLens[depIdx]);
                    std::wstring depFilename = darkroom::getWstrFromUtf8(depFilenameUTF8);

                    if (PathIsRelativeW(depFilename.c_str()))
                    {
                        depFilename = rootDir + depFilename;
                    }

                    if (!matchFileTime(depFilename.c_str(), depTimestamp, true))
                    {
                        needRecompilation = true;
                        break;
                    }
                }
            }
        }

        if (needRecompilation && IsEffectHardcoded(effectNameNoExt))
        {
            WriteHardcodedBinaryFile(effectNameNoExt, binaryPathAndName);
            if (ir::filehelpers::validateFileExists(binaryPathAndName))
            {
                needRecompilation = false;
            }
        }

        if (needRecompilation)
        {
            // TODO: avoid tool invocation, if the file (and its dependencies) passes dates requirements
            HANDLE hndl = 0;
            std::wstring errorFilepath = binaryPathOnly + L"\\" + effectNameNoExt + L"_error.txt";
            std::wstring args = L"--input-path \"" + effectName + L"\" --output-path \"" + binaryPathAndName + L"\" --error-file \"" + errorFilepath + L"\"";
            exelwteProcess(L"\"" + m_fxToolFilepath + L"\"", { args }, m_rootDir, hndl);

            // Wait for the process to finish
            DWORD wfso;
            do
            {
                wfso = WaitForSingleObject(hndl, 1);
            } while (wfso != WAIT_ABANDONED && wfso != WAIT_OBJECT_0 && wfso != WAIT_FAILED);

            DWORD exitCode = -1;
            if (wfso != WAIT_FAILED)
            {
                GetExitCodeProcess(hndl, &exitCode);
            }

            CloseHandle(hndl);

            if (exitCode != 0)
            {
                std::ifstream input(errorFilepath.c_str());
                std::string error_line;

                if (input.is_open())
                {
                    std::getline(input, error_line);
                }

                if (error_line.length() > 0)
                {
                    throw MultipassConfigParserError(
                        MultipassConfigParserErrorEnum::eInternalError,
                        "Effect compiler failure: " + error_line
                        );
                }
                else
                {
                    throw MultipassConfigParserError(
                        MultipassConfigParserErrorEnum::eInternalError,
                        "Effect compiler tool reported failure"
                        );
                }
            }
        }

        acefEffectStorageHeaderCheck.freeAllocatedMem();

        bool calcHashes = (pExpectedHashSet != nullptr);

        // Hashing scheme:
        //  - hash ACEF binary, initial salt "acef10", shifted by +1
        //  - successive hash shaders binaries:
        //      * initial salt: previous ACEF hash, plus "shbin", shifted by +2
        //      * first, cumulative hash of VS shader binaries, retrieved from the ir::Effect::finalize()
        //      * second, cumulative hash of VS shader binaries, retrieved from the ir::Effect::finalize()
        //  - hash resources:
        //      * initial salt: previous shader bin hash, plus "txres", shifted by -1

        auto finalizeSalt = [](const int * inSalt, uint8_t * outSalt, size_t saltSize, int offset)
        {
            for (size_t si = 0; si < saltSize; ++si)
            {
                outSalt[si] = (uint8_t)(inSalt[si]+offset);
            }
        };

        // Load the file to callwlate hashes (if required)
        bool acefHashValid = false;
        if (calcHashes)
        {
            FILE * fp = nullptr;
            _wfopen_s(&fp, binaryPathAndName.c_str(), L"rb");

            if (!fp)
            {
                throw MultipassConfigParserError(
                    MultipassConfigParserErrorEnum::eInternalError,
                    "File not found!"
                    );
            }

            fseek(fp, 0, SEEK_END);
            long binaryFileSize = ftell(fp);
            fseek(fp, 0, SEEK_SET);

            std::vector<unsigned char> fileBinaryContents;
            fileBinaryContents.resize(binaryFileSize);

            fread(fileBinaryContents.data(), binaryFileSize, 1, fp);

            fclose(fp);

            // We don't care about timestamps, so zero them out for hash callwlation
            acef::eraseTimestamps(fileBinaryContents.data());

            const size_t acefHashSaltSize = 6;
            int acefHashSalt[acefHashSaltSize] =
                { (int)'a', (int)'c', (int)'e', (int)'f', (int)'1', (int)'0' };

            uint8_t acefHashSaltFinal[acefHashSaltSize];
            finalizeSalt(acefHashSalt, acefHashSaltFinal, acefHashSaltSize, 1);

            sha256_ctx hashState;
            sha256_init(&hashState);
            sha256_update(&hashState, acefHashSaltFinal, (uint32_t)acefHashSaltSize);
            sha256_update(&hashState, (uint8_t *)fileBinaryContents.data(), (uint32_t)binaryFileSize);
            m_callwlatedHashes.UpdateHash(&hashState, Hash::Type::Acef);

            acefHashValid = true;
        }

        // LOADING
        acef::EffectStorage acefEffectStorageLoaded;
        acef::load(binaryPathAndName.c_str(), &acefEffectStorageLoaded);

        // ACEF->IR colwersion
        /////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////

        m_irEffect.reset();

        bool doesEffectRequireColor = false;
        m_effectRequiresDepth = false;
        m_effectRequiresHUDless = false;
        m_effectRequiresHDR = false;
        m_effectRequiresColorBase = false;

        wchar_t config_path[FILENAME_MAX];
        {
            size_t pathEndingPos;
            swprintf_s(config_path, FILENAME_MAX, L"%s%s", rootDir, fxFilename);

            for (pathEndingPos = wcslen(config_path) - 1;
                 pathEndingPos > 0 && config_path[pathEndingPos] != L'\\' && config_path[pathEndingPos] != L'/';
                 --pathEndingPos)
            {
            }

            if (pathEndingPos != 0)
                config_path[pathEndingPos] = 0;
        }
        const size_t config_path_len = wcslen(config_path);

        ir::Texture * outTex = nullptr;

        try
        {
            // m_irEffect will contain hashes, if required
            MultipassConfigParserError err = shadermod::ir::colwertBinaryToIR(
                // in
                rootDir, tempsDir,

                finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput,

                config_path, binaryPathOnly.c_str(), acefEffectStorageLoaded,
                // out
                &m_irEffect, &doesEffectRequireColor, &m_effectRequiresDepth,
                &m_effectRequiresHUDless, &m_effectRequiresHDR,
                &m_effectRequiresColorBase,
                &outTex,
                calcHashes
                );

            if (err)
                throw err;
        }
        catch (const MultipassConfigParserError& err)
        {
            throw err;
        }

        // Join hashes
        if (calcHashes)
        {
            if ((!m_irEffect.m_hashesValid || !acefHashValid) && compareHashes)
            {
                throw MultipassConfigParserError(
                    MultipassConfigParserErrorEnum::eInternalError,
                    ""
                );
            }

            const size_t shbinHashSaltSize = 5;
            int shbinHashSalt[shbinHashSaltSize] =
            { (int)'s', (int)'h', (int)'b', (int)'i', (int)'n' };

            uint8_t shbinHashSaltFinal[shbinHashSaltSize];
            finalizeSalt(shbinHashSalt, shbinHashSaltFinal, shbinHashSaltSize, 2);

            sha256_ctx hashState;
            sha256_init(&hashState);
            sha256_update(&hashState, getCallwlatedACEFBinaryHash(), SHA256_DIGEST_SIZE);
            sha256_update(&hashState, shbinHashSaltFinal, (uint32_t)shbinHashSaltSize);
            sha256_update(&hashState, (uint8_t *)m_irEffect.m_vsBlobsHash, (uint32_t)ir::Effect::c_hashKeySizeBytes);
            sha256_update(&hashState, (uint8_t *)m_irEffect.m_psBlobsHash, (uint32_t)ir::Effect::c_hashKeySizeBytes);
            m_callwlatedHashes.UpdateHash(&hashState, Hash::Type::Shader);

            const size_t txresHashSaltSize = 5;
            int txresHashSalt[txresHashSaltSize] =
            { (int)'t', (int)'x', (int)'r', (int)'e', (int)'s' };

            uint8_t txresHashSaltFinal[txresHashSaltSize];
            finalizeSalt(txresHashSalt, txresHashSaltFinal, txresHashSaltSize, -2);

            sha256_init(&hashState);
            sha256_update(&hashState, getCallwlatedShaderHash(), SHA256_DIGEST_SIZE);
            sha256_update(&hashState, txresHashSaltFinal, (uint32_t)txresHashSaltSize);
            sha256_update(&hashState, (uint8_t *)m_irEffect.m_texturesHash, (uint32_t)ir::Effect::c_hashKeySizeBytes);
            m_callwlatedHashes.UpdateHash(&hashState, Hash::Type::Resource);

            if (compareHashes)
            {
                if (pExpectedHashSet->find(m_callwlatedHashes) == pExpectedHashSet->end())
                {
                    throw MultipassConfigParserError(
                        MultipassConfigParserErrorEnum::eInternalError,
                        "ACEF hash check fail"
                    );
                }

                m_expectedHashSet = *pExpectedHashSet;
            }
        }

        m_effect.m_outputColorTex = outTex->m_D3DTexture;
        m_outputFormat = outTex->m_format;
        m_outputWidth = outTex->m_width;
        m_outputHeight = outTex->m_height;

        if (doesEffectRequireColor)
            m_effect.m_inputColorTex = finalColorInput.texture;

        if (m_effectRequiresDepth)
            m_effect.m_inputDepthTex = depthInput.texture;

        /////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////

        acefEffectStorageLoaded.freeAllocatedMem();

        return;
    }

    MultiPassEffect::MultiPassEffect(
            const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,
            const std::map<std::wstring, std::wstring> & fxExtensionToolMap,

            const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
            const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
            const ir::Effect::InputData & colorBaseInput,

            ID3D11Device* d3dDevice, D3DCompilerHandler* d3dCompiler,

            const std::set<Hash::Effects> * pExpectedHashSet,
            bool compareHashes
            ):
                m_resourceManager(d3dDevice, d3dCompiler),
                m_irEffect(&m_effect, &m_resourceManager),
                m_systemConstantsNeverSet(true),
                m_rootDir(rootDir),
                m_tempsDir(tempsDir),
                m_fxFilename(fxFilename)
    {
        size_t effectNameLen = wcslen(fxFilename);

        // Extension without the dot
        const size_t extPos = m_fxFilename.find_last_of(L'.');
        if (extPos == std::wstring::npos)
        {
            MultipassConfigParserError err = MultipassConfigParserError(
                MultipassConfigParserErrorEnum::eInternalError,
                "No FX extension"
                );

            throw err;
        }

        const wchar_t * extension = m_fxFilename.c_str() + extPos + 1;
        auto fxToolPair = fxExtensionToolMap.find(std::wstring(extension));
        if (fxToolPair == fxExtensionToolMap.end())
        {
            MultipassConfigParserError err = MultipassConfigParserError(
                MultipassConfigParserErrorEnum::eInternalError,
                "Unknown FX extension"
                );

            throw err;
        }

        std::wstring fxToolFilepath = std::wstring(installDir) + fxToolPair->second;
        initializeEffect(
            fxToolFilepath.c_str(), rootDir, tempsDir, fxFilename,
            finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput,
            pExpectedHashSet,
            compareHashes
            );
    }

    MultiPassEffect::MultiPassEffect(
            const wchar_t * fxToolFilepath, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,

            const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
            const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
            const ir::Effect::InputData & colorBaseInput,

            ID3D11Device* d3dDevice, D3DCompilerHandler* d3dCompiler,

            const std::set<Hash::Effects> * pExpectedHashSet,
            bool compareHashes
            ):
                m_resourceManager(d3dDevice, d3dCompiler),
                m_irEffect(&m_effect, &m_resourceManager),
                m_systemConstantsNeverSet(true),
                m_rootDir(rootDir),
                m_tempsDir(tempsDir),
                m_fxFilename(fxFilename)
    {
        initializeEffect(
            fxToolFilepath, rootDir, tempsDir, fxFilename,
            finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput,
            pExpectedHashSet,
            compareHashes
            );
    }

    MultiPassEffect::~MultiPassEffect()
    {
    }

    const ir::UserConstantManager& MultiPassEffect::getUserConstantManager() const
    {
        return m_irEffect.getUserConstantManager();
    }

    ir::UserConstantManager& MultiPassEffect::getUserConstantManager()
    {
        return m_irEffect.getUserConstantManager();
    }

    bool MultiPassEffect::isForceUpdateOfSystemConstantsNeeded() const
    {
        return m_systemConstantsNeverSet;
    }

    void MultiPassEffect::markSystemConstantsUpdatedOnce()
    {
        m_systemConstantsNeverSet = false;
    }

    ir::FragmentFormat  MultiPassEffect::getOutFormat() const
    {
        return m_outputFormat;
    }

    ID3D11Texture2D* MultiPassEffect::getOutputColorTexture() const
    {
        return m_effect.m_outputColorTex;
    }

    const std::wstring& MultiPassEffect::getFxToolFilepath() const
    {
        return m_fxToolFilepath;
    }

    const std::wstring& MultiPassEffect::getRootDir() const
    {
        return m_rootDir;
    }

    const std::wstring& MultiPassEffect::getTempsDir() const
    {
        return m_tempsDir;
    }

    const std::wstring& MultiPassEffect::getFxFilename() const
    {
        return m_fxFilename;
    }

    const std::wstring& MultiPassEffect::getFxFileFullPath() const
    {
        return m_fxFilepath;
    }

    unsigned int MultiPassEffect::getOutputWidth() const
    {
        return m_outputWidth;
    }

    unsigned int MultiPassEffect::getOutputHeight() const
    {
        return m_outputHeight;
    }

    const CmdProcEffect& MultiPassEffect::getLowLevelEffect() const
    {
        return m_effect;
    }

    const uint8_t * MultiPassEffect::getCallwlatedShaderHash() const    { return m_callwlatedHashes.GetHash(Hash::Type::Shader).data(); }
    const uint8_t * MultiPassEffect::getCallwlatedResourceHash() const  { return m_callwlatedHashes.GetHash(Hash::Type::Resource).data(); }
    const uint8_t * MultiPassEffect::getCallwlatedACEFBinaryHash() const{ return m_callwlatedHashes.GetHash(Hash::Type::Acef).data(); }

    bool MultiPassEffect::changeInputs(
        const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
        const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
        const ir::Effect::InputData & colorBaseInput,
        bool ignoreTexturesNotSet
        )
    {
        ID3D11Texture2D* oldColorSourceTexture = m_irEffect.getInputColorTexture();
        ID3D11Texture2D* oldDepthSourceTexture = m_irEffect.getInputDepthTexture();
        ID3D11Texture2D* oldHUDlessSourceTexture = m_irEffect.getInputHUDlessTexture();
        ID3D11Texture2D* oldHDRSourceTexture = m_irEffect.getInputHDRTexture();
        ID3D11Texture2D* oldColorBaseTexture = m_irEffect.getInputColorBaseTexture();

        if (finalColorInput.texture == oldColorSourceTexture && depthInput.texture == oldDepthSourceTexture &&
            hudlessInput.texture == oldHUDlessSourceTexture && hdrInput.texture == oldHDRSourceTexture &&
            colorBaseInput.texture == oldColorBaseTexture)
            return true;

        if (m_irEffect.isInputColorTextureSet() != (finalColorInput.texture != nullptr))
            return false;

        if (finalColorInput.texture)
        {
            assert(finalColorInput.format != ir::FragmentFormat::kNUM_ENTRIES);
            if (finalColorInput.format != m_irEffect.getInputColorFormat())
                return false;

            if (finalColorInput.width != m_irEffect.getInputWidth() || finalColorInput.height != m_irEffect.getInputHeight())
                return false;
        }

        if (!ignoreTexturesNotSet)
        {
            if (m_irEffect.isInputDepthTextureSet() != (depthInput.texture != nullptr))
                return false;
        }

        if (depthInput.texture)
        {
            assert(depthInput.format != ir::FragmentFormat::kNUM_ENTRIES);
            if (depthInput.format != m_irEffect.getInputDepthFormat())
                return false;

            if (depthInput.width != m_irEffect.getInputDepthWidth() || depthInput.height != m_irEffect.getInputDepthHeight())
                return false;
        }

        if (m_effectRequiresHUDless)
        {
            if (!ignoreTexturesNotSet)
            {
                if (m_irEffect.isInputHUDlessTextureSet() != (hudlessInput.texture != nullptr))
                    return false;
            }

            if (hudlessInput.texture)
            {
                assert(hudlessInput.format != ir::FragmentFormat::kNUM_ENTRIES);
                if (hudlessInput.format != m_irEffect.getInputHUDlessFormat())
                    return false;

                if (hudlessInput.width != m_irEffect.getInputHUDlessWidth() || hudlessInput.height != m_irEffect.getInputHUDlessHeight())
                    return false;
            }
        }

        if (m_effectRequiresHDR)
        {
            if (!ignoreTexturesNotSet)
            {
                if (m_irEffect.isInputHDRTextureSet() != (hdrInput.texture != nullptr))
                    return false;
            }

            if (hdrInput.texture)
            {
                assert(hdrInput.format != ir::FragmentFormat::kNUM_ENTRIES);
                if (hdrInput.format != m_irEffect.getInputHDRFormat())
                    return false;

                if (hdrInput.width != m_irEffect.getInputHDRWidth() || hdrInput.height != m_irEffect.getInputHDRHeight())
                    return false;
            }
        }

        if (m_effectRequiresColorBase)
        {
            if (m_irEffect.isInputColorBaseTextureSet() != (colorBaseInput.texture != nullptr))
                return false;

            if (colorBaseInput.texture)
            {
                assert(colorBaseInput.format != ir::FragmentFormat::kNUM_ENTRIES);
                if (colorBaseInput.format != m_irEffect.getInputColorBaseFormat())
                    return false;

                if (colorBaseInput.width != m_irEffect.getInputColorBaseWidth() || colorBaseInput.height != m_irEffect.getInputColorBaseHeight())
                    return false;
            }
        }

        m_irEffect.setInputs(finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput);

        m_irEffect.fixInputs();

        return true;
    }

    MultiPassProcessor::MultiPassProcessor(const wchar_t * compilerPath, bool doInitFramework, LARGE_INTEGER adapterLUID):
        m_isValid(false)
    {
        m_d3dCompiler.setD3DCompilerPath(compilerPath);

        if (doInitFramework)
            initFramework(adapterLUID);
    }

    MultiPassProcessor::MultiPassProcessor(bool doInitFramework, LARGE_INTEGER adapterLUID):
        m_isValid(false)
    {
        m_d3dCompiler.setD3DCompilerPath(L"d3dcompiler_47.dll");

        if (doInitFramework)
            initFramework(adapterLUID);
    }

    MultiPassProcessor::~MultiPassProcessor()
    {
        destroyAllEffectsInStack();
        destroyDevice();
    }

    void MultiPassProcessor::setD3DCompilerPath(const wchar_t * compilerPath)
    {
        m_d3dCompiler.setD3DCompilerPath(compilerPath);
    }

    bool MultiPassProcessor::initFramework(LARGE_INTEGER adapterLUID)
    {
        assert(!m_isValid);

        HRESULT hr = S_OK;
        UINT createDeviceFlags = 0;
#ifdef _DEBUG
        createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        D3D_FEATURE_LEVEL featureLevels[] =
        {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        };

        UINT numFeatureLevels = ARRAYSIZE(featureLevels);

        IDXGIFactory1* dxgiFactory1 = nullptr;
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)(&dxgiFactory1));
        if (FAILED(hr))
        {
            return false;
        }

#define LWIDIA_VENDOR_ID    0x10DE

        DXGI_ADAPTER_DESC adapterDesc;
        bool foundLwDevice = false;
        UINT i = 0;
        while (dxgiFactory1->EnumAdapters1(i, &m_pAdapter) != DXGI_ERROR_NOT_FOUND)
        {
            hr = m_pAdapter->GetDesc(&adapterDesc);
            assert(SUCCEEDED(hr));

            bool luidMatch =
                (adapterLUID.QuadPart == 0) ||
                ((adapterDesc.AdapterLuid.HighPart == adapterLUID.HighPart) &&
                (adapterDesc.AdapterLuid.LowPart == adapterLUID.LowPart));

            if ((adapterDesc.VendorId == LWIDIA_VENDOR_ID) && luidMatch)
            {
                foundLwDevice = true;
                break;
            }
            i++;
        }

#undef LWIDIA_VENDOR_ID

        D3D_FEATURE_LEVEL * lwrrentFeatureLevel = featureLevels;
        UINT lwrrentNumFeatureLevels = numFeatureLevels;

        // From MSDN:
        //  If you set the pAdapter parameter to a non-NULL value, you must also set the
        //  DriverType parameter to the D3D_DRIVER_TYPE_UNKNOWN value. If you set the pAdapter
        //  parameter to a non-NULL value and the DriverType parameter to th
        //  D3D_DRIVER_TYPE_HARDWARE value, D3D11CreateDevice returns an HRESULT of E_ILWALIDARG
        m_driverType = D3D_DRIVER_TYPE_UNKNOWN;

        // We have several possible expected failure cases, which do not mean error actually
        //  example: device fails with E_ILWALIDARG, we need to remove D3D_FEATURE_LEVEL_11_1 and then
        //  it fails because there is no graphics debugging, but if we'll remove debug layer request
        //  device will be successfully created
        int deviceCreationHP = 10;
        while (deviceCreationHP > 0)
        {
            hr = D3D11CreateDevice(m_pAdapter, m_driverType, nullptr, createDeviceFlags, lwrrentFeatureLevel, lwrrentNumFeatureLevels,
                D3D11_SDK_VERSION, &m_d3dDevice, &m_featureLevel, &m_immediateContext);

            --deviceCreationHP;

            // Only try to remove D3D_FEATURE_LEVEL_11_1 if it is still present
            if (hr == E_ILWALIDARG && (lwrrentNumFeatureLevels == numFeatureLevels))
            {
                // E_ILWALIDARG is expected if feature level D3D_FEATURE_LEVEL_11_1 is not supported
                // From MSDN:
                //  If you provide a D3D_FEATURE_LEVEL array that contains D3D_FEATURE_LEVEL_11_1 on a computer that
                //  doesn't have the Direct3D 11.1 runtime installed, this function immediately fails with E_ILWALIDARG
                lwrrentFeatureLevel = featureLevels + 1;
                lwrrentNumFeatureLevels = numFeatureLevels - 1;
            }
            else if (hr == DXGI_ERROR_SDK_COMPONENT_MISSING)
            {
                // The debug layer might not be installed, so we need to retry without it
                createDeviceFlags &= ~D3D11_CREATE_DEVICE_DEBUG;
                LOG_INFO("D3D11 Debug device was requested, but debug layer is not installed");
            }
            else
            {
                // In case of failures we don't know how to process, or in case of success
                deviceCreationHP = 0;
            }
        }

        CHECK_RESULT(hr, "failed to initialize device");

        dxgiFactory1->Release();

        // Create the vertex shader
        hr = m_d3dDevice->CreateVertexShader(g_drawFullScreenTriangleVS, sizeof(g_drawFullScreenTriangleVS), nullptr, &m_vertexShader);
        CHECK_RESULT(hr, "failed to init default vertex shader");

        m_depthSourceTexture = nullptr;
        m_colorSourceTexture = nullptr;

        m_colorSourceTextureFormat = ir::FragmentFormat::kNUM_ENTRIES;
        m_depthSourceTextureFormat = ir::FragmentFormat::kNUM_ENTRIES;

        m_width = undefinedSize;
        m_height = undefinedSize;

        m_timer.Start();

        m_dt = 0.0;
        m_elapsedTime = 0.0;

        if (!createCopyShader())
            return false;

        m_isValid = true;

        CheckLwbinSupportOnHW(m_d3dDevice);

        m_SharpenLwbinData.m_SharpenParameter = 0.0f;
        m_SharpenLwbinData.m_DenoiseParameter = 0.0f;

        return true;
    }

    void MultiPassProcessor::destroyDevice()
    {
        m_isValid = false;

        destroyOutSRV();
        destroyCopyShader();

        if (m_immediateContext)
        {
            m_immediateContext->ClearState();
            m_immediateContext->Flush();
        }

        SAFE_RELEASE(m_vertexShader);
        SAFE_RELEASE(m_immediateContext);

#if _DEBUG
        if (m_d3dDevice)
        {
            ID3D11Debug* DebugDevice = nullptr;
            HRESULT Result = m_d3dDevice->QueryInterface(__uuidof(ID3D11Debug), reinterpret_cast<void**>(&DebugDevice));
            if (DebugDevice)
            {
                Result = DebugDevice->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
                SAFE_RELEASE(DebugDevice);
            }
        }
#endif
        SAFE_RELEASE(m_d3dDevice);
    }

    MultiPassEffect* MultiPassProcessor::createEffectFromACEF(
                        const wchar_t * fxToolFilepath, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,

                        const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
                        const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
                        const ir::Effect::InputData & colorBaseInput,

                        MultipassConfigParserError& err,

                        const std::set<Hash::Effects> * pExpectedHashSet,
                        bool compareHashes
                        )
    {
        MultiPassEffect* ret = nullptr;

        try
        {
            ret = m_effectsPool.newElement(fxToolFilepath, rootDir, tempsDir, fxFilename,
                                    finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput,
                                    m_d3dDevice, &m_d3dCompiler, pExpectedHashSet, compareHashes);
        }
        catch (const MultipassConfigParserError& e)
        {
            err = e;

            return nullptr;
        }

        return ret;
    }

    MultiPassEffect* MultiPassProcessor::createEffectFromACEF(
                        const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * fxFilename,
                        const std::map<std::wstring, std::wstring> & fxExtensionToolMap,

                        const ir::Effect::InputData & finalColorInput, const ir::Effect::InputData & depthInput,
                        const ir::Effect::InputData & hudlessInput, const ir::Effect::InputData & hdrInput,
                        const ir::Effect::InputData & colorBaseInput,

                        MultipassConfigParserError& err,

                        const std::set<Hash::Effects> * pExpectedHashSet,
                        bool compareHashes
                        )
    {
        MultiPassEffect* ret = nullptr;

        try
        {
            ret = m_effectsPool.newElement(installDir, rootDir, tempsDir, fxFilename,
                                            fxExtensionToolMap,
                                            finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput,
                                            m_d3dDevice, &m_d3dCompiler, pExpectedHashSet, compareHashes);
        }
        catch (const MultipassConfigParserError& e)
        {
            err = e;

            return nullptr;
        }

        return ret;
    }

    void MultiPassProcessor::destroyEffect(MultiPassEffect* eff)
    {
        m_effectsPool.deleteElement(eff);
    }

    unsigned int MultiPassProcessor::getNumEffects() const
    {
        return (unsigned int) m_effectsStackInternal.size();
    }

    MultiPassEffect* MultiPassProcessor::getEffect(unsigned int idx)
    {
        if (idx < m_effectsStackInternal.size())
            return m_effectsStackInternal[idx];
        else
            return nullptr;
    }

    MultiPassEffect* MultiPassProcessor::getEffect(unsigned int idx) const
    {
        if (idx < m_effectsStackInternal.size())
            return m_effectsStackInternal[idx];
        else
            return nullptr;
    }

#if 0 //better not use those methods as they woul leave your stak in a dangling state if an effect down the pipe fails the compilation
    MultipassConfigParserError MultiPassProcessor::insertEffect(MultiPassEffect* eff, unsigned int idx)
    {
        MultipassConfigParserError err(MultipassConfigParserErrorEnum::eOK);

        assert(idx <= m_effectsStackInternal.size());

        m_effectsStackInternal.insert(m_effectsStackInternal.begin() + idx, eff);

        err = rebuildEffectsInStack(idx);
        onLastEffectUpdated(!err && areInputsValid());

        return err;
    }

    MultipassConfigParserError MultiPassProcessor::eraseEffect(unsigned int idx, bool dontDestroy = false)
    {
        MultipassConfigParserError err(MultipassConfigParserErrorEnum::eOK);

        assert(idx < m_effectsStackInternal.size());

        if (!dontDestroy)
            destroyEffect(m_effectsStackInternal[idx]);

        m_effectsStackInternal.erase(m_effectsStackInternal.begin() + idx);

        err = rebuildEffectsInStack(idx);
        onLastEffectUpdated(!err && areInputsValid());

        return err;
    }
#endif

    void MultiPassProcessor::pushBackEffect(MultiPassEffect* eff)
    {
        m_effectsStackInternal.push_back(eff);
        onLastEffectUpdated(true);
    }

    MultipassConfigParserError MultiPassProcessor::replaceEffect(
                    const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir,
                    const std::map<std::wstring, std::wstring> & fxExtensionToolMap,
                    const wchar_t * fxFilename, MultiPassEffect ** effectPtr, int stackIdx,

                    const std::set<Hash::Effects> * pExpectedHashSet,
                    bool compareHashes
                    )
    {
        MultipassConfigParserError err(MultipassConfigParserErrorEnum::eOK);

        // stackIdx == [0..size()-1] - replace codeptah
        // stackIdx == -1 - pushback codepath
        if (stackIdx >= (int)m_effectsStackInternal.size())
            return MultipassConfigParserErrorEnum::eInternalError;

        if (stackIdx >= 0)
        {
            destroyEffect(m_effectsStackInternal[stackIdx]);
            m_effectsStackInternal[stackIdx] = nullptr;
        }

        if (effectPtr)
            *effectPtr = nullptr;

        ir::Effect::InputData finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput;

        finalColorInput.width = m_width;
        finalColorInput.height = m_height;
        finalColorInput.format = m_colorSourceTextureFormat;
        finalColorInput.texture = m_colorSourceTexture;

        depthInput.width = m_depthWidth;
        depthInput.height = m_depthHeight;
        depthInput.format = m_depthSourceTextureFormat;
        depthInput.texture = m_depthSourceTexture;

        hudlessInput.width = m_hudlessWidth;
        hudlessInput.height = m_hudlessHeight;
        hudlessInput.format = m_hudlessSourceTextureFormat;
        hudlessInput.texture = m_hudlessSourceTexture;

        hdrInput.width = m_hdrWidth;
        hdrInput.height = m_hdrHeight;
        hdrInput.format = m_hdrSourceTextureFormat;
        hdrInput.texture = m_hdrSourceTexture;

        colorBaseInput.width = m_width;
        colorBaseInput.height = m_height;
        colorBaseInput.format = m_colorSourceTextureFormat;
        colorBaseInput.texture = m_colorSourceTexture;

        int prevEffectIdx = -1;
        if (stackIdx > 0)
        {
            // Replace codepath
            prevEffectIdx = stackIdx - 1;
        }
        else if (stackIdx < 0)
        {
            // PushBack codepath
            prevEffectIdx = (int)m_effectsStackInternal.size() - 1;
        }

        if (prevEffectIdx >= 0)
        {
            MultiPassEffect* eff = m_effectsStackInternal[prevEffectIdx];

            finalColorInput.width = eff->getOutputWidth();
            finalColorInput.height = eff->getOutputHeight();
            finalColorInput.format = eff->getOutFormat();
            finalColorInput.texture = eff->getOutputColorTexture();
        }

        MultiPassEffect* eff = createEffectFromACEF(
                    installDir, rootDir, tempsDir, fxFilename,
                    fxExtensionToolMap,
                    finalColorInput,
                    depthInput,
                    hudlessInput,
                    hdrInput,
                    colorBaseInput,
                    err,
                    pExpectedHashSet,
                    compareHashes
                    );

        if (err)
            return err;

        if (effectPtr)
            *effectPtr = eff;

        if (stackIdx >= 0)
        {
            // Replace codepath
            m_effectsStackInternal[stackIdx] = eff;
        }
        else
        {
            // PushBack codepath
            pushBackEffect(eff);
        }

        return err;
    }

    MultipassConfigParserError MultiPassProcessor::removeSingleEffect(unsigned int idx, bool dontDestroy)
    {
        MultipassConfigParserError err(MultipassConfigParserErrorEnum::eOK);

        assert(idx < m_effectsStackInternal.size());

        if (!dontDestroy)
            destroyEffect(m_effectsStackInternal[idx]);

        m_effectsStackInternal.erase(m_effectsStackInternal.begin() + idx);

        return err;
    }

    MultipassConfigParserError MultiPassProcessor::pushBackEffect(
            const wchar_t * installDir, const wchar_t * rootDir, const wchar_t * tempsDir,
            const std::map<std::wstring, std::wstring> & fxExtensionToolMap,
            const wchar_t * fxFilename, MultiPassEffect ** effectPtr,
            const std::set<Hash::Effects> * pExpectedHashSet,
            bool compareHashes
            )
    {
        return replaceEffect(installDir, rootDir, tempsDir, fxExtensionToolMap, fxFilename, effectPtr, -1, pExpectedHashSet, compareHashes);
    }

    void MultiPassProcessor::popBackEffect(bool dontDestroy)
    {
        if (!dontDestroy)
            destroyEffect(m_effectsStackInternal.back());

        m_effectsStackInternal.pop_back();

        onLastEffectUpdated(true);
    }

    // By using this function, user guarantees that surface parameters didn't change for replaced effects
    MultipassConfigParserError MultiPassProcessor::relinkEffects(bool ignoreTexturesNotSet)
    {
        MultipassConfigParserError err(MultipassConfigParserErrorEnum::eOK);

        ir::Effect::InputData finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput;

        finalColorInput.width = m_width;
        finalColorInput.height = m_height;
        finalColorInput.format = m_colorSourceTextureFormat;
        finalColorInput.texture = m_colorSourceTexture;

        depthInput.width = m_depthWidth;
        depthInput.height = m_depthHeight;
        depthInput.format = m_depthSourceTextureFormat;
        depthInput.texture = m_depthSourceTexture;

        hudlessInput.width = m_hudlessWidth;
        hudlessInput.height = m_hudlessHeight;
        hudlessInput.format = m_hudlessSourceTextureFormat;
        hudlessInput.texture = m_hudlessSourceTexture;

        hdrInput.width = m_hdrWidth;
        hdrInput.height = m_hdrHeight;
        hdrInput.format = m_hdrSourceTextureFormat;
        hdrInput.texture = m_hdrSourceTexture;

        colorBaseInput.width = m_width;
        colorBaseInput.height = m_height;
        colorBaseInput.format = m_colorSourceTextureFormat;
        colorBaseInput.texture = m_colorSourceTexture;

        const size_t numEffectsOnStack = m_effectsStackInternal.size();

        MultiPassEffect* eff;
        for (size_t effIdx = 0; effIdx < numEffectsOnStack; ++effIdx)
        {
            eff = m_effectsStackInternal[effIdx];
            if (!eff->changeInputs(finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput, ignoreTexturesNotSet))
            {
                // Some of the sizes changed, need whole stack rebuilt
                return MultipassConfigParserErrorEnum::eInternalError;
            }

            finalColorInput.width = eff->getOutputWidth();
            finalColorInput.height = eff->getOutputHeight();
            finalColorInput.format = eff->getOutFormat();
            finalColorInput.texture = eff->getOutputColorTexture();
        }
        onLastEffectUpdated(true);

        return err;
    }

    //it's the user's duty to destroy all effects that were created and aren't in the stack - no automatic cleanup here!
    void MultiPassProcessor::destroyAllEffectsInStack()
    {
        for (auto effect : m_effectsStackInternal)
        {
            destroyEffect(effect);
        }

        m_effectsStackInternal.clear();
        m_isDefunctEffectInStack = false;
    }

    void MultiPassProcessor::onLastEffectUpdated(bool updateSucceeded)
    {
        if (updateSucceeded)
            updateOutSRV();
        else
            destroyOutSRV();
    }

    CmdProcConstDataType MultiPassProcessor::getConstantDataType(CmdProcConstHandle h, const MultiPassEffect *eff) const
    {
        const ir::UserConstantManager& ucm = eff->getUserConstantManager();

        if (!isSystemConst(h))
        {
            const ir::UserConstant * lwrUC = ucm.getUserConstantByIndex(toUserConstIndex(h));
            return ir::ircolwert::userConstTypeToCmdProcConstElementDataType(lwrUC->getType());
        }
        else
        {
            return getCmdProcSystemConstElementDataType(toSystemConst(h));
        }
    }

    unsigned int MultiPassProcessor::getConstantDataDimensions(CmdProcConstHandle h, const MultiPassEffect *eff) const
    {
        const ir::UserConstantManager& ucm = eff->getUserConstantManager();

        if (!isSystemConst(h))
        {
            const ir::UserConstant * lwrUC = ucm.getUserConstantByIndex(toUserConstIndex(h));
            return lwrUC->getDefaultValue().getDimensionality();
        }
        else
        {
            return getCmdProcSystemConstDimensions(toSystemConst(h));
        }
    }

    bool MultiPassProcessor::writeConstantValue(CmdProcConstHandle h, const MultiPassEffect *eff, void* buf, size_t bytesToCopy) const
    {
        const ir::UserConstantManager& ucm = eff->getUserConstantManager();

        if (!isSystemConst(h))
        {
            return ucm.getConstantValue(toUserConstIndex(h), buf, bytesToCopy);
        }
        else
        {
            const ir::userConstTypes::Float cDT = (float)(m_dt);
            const ir::userConstTypes::Float cElapsedTime = (float)(m_elapsedTime * 1000.0);

            CmdProcSystemConst sch = toSystemConst(h);

            size_t sz = getCmdProcConstDataElementTypeSize(getCmdProcSystemConstElementDataType(sch));
            bool ok = bytesToCopy >= sz;

            if (!ok)
                return false;

            switch (sch)
            {
            case CmdProcSystemConst::kDT:
            {
                *reinterpret_cast<ir::userConstTypes::Float *>(buf) = cDT;
                break;
            }
            case CmdProcSystemConst::kElapsedTime:
            {
                *reinterpret_cast<ir::userConstTypes::Float *>(buf) = cElapsedTime;
                break;
            }
            case CmdProcSystemConst::kFrame:
            {
                *reinterpret_cast<ir::userConstTypes::Int *>(buf) = m_processingFrameNum;
                break;
            }
            case CmdProcSystemConst::kScreenSize:
            {
                *reinterpret_cast<ir::userConstTypes::Float *>(buf) = (float)m_width;
                *(reinterpret_cast<ir::userConstTypes::Float *>(buf)+1) = (float)m_height;
                break;
            }
            case CmdProcSystemConst::kCaptureState:
            {
                *reinterpret_cast<ir::userConstTypes::Int *>(buf) = m_captureState;
                break;
            }
            case CmdProcSystemConst::kTileUV:
            {
                // Constant spans region top-left to bottom-right
                *reinterpret_cast<ir::userConstTypes::Float *>(buf) = m_tileInfo.m_tileTLU;
                *(reinterpret_cast<ir::userConstTypes::Float *>(buf)+1) = m_tileInfo.m_tileTLV;
                *(reinterpret_cast<ir::userConstTypes::Float *>(buf)+2) = m_tileInfo.m_tileBRU;
                *(reinterpret_cast<ir::userConstTypes::Float *>(buf)+3) = m_tileInfo.m_tileBRV;
                break;
            }
            case CmdProcSystemConst::kDepthAvailable:
            {
                *reinterpret_cast<ir::userConstTypes::Int *>(buf) = (int)m_isDepthAvailableShader;
                break;
            }
            case CmdProcSystemConst::kHDRAvailable:
            {
                *reinterpret_cast<ir::userConstTypes::Int *>(buf) = (int)m_isHDRAvailableShader;
                break;
            }
            case CmdProcSystemConst::kHUDlessAvailable:
            {
                *reinterpret_cast<ir::userConstTypes::Int *>(buf) = (int)m_isHUDlessAvailableShader;
                break;
            }
            default:
                return false;
            }
        }

        return true;
    }

    bool MultiPassProcessor::isConstantDirty(CmdProcConstHandle h, const MultiPassEffect *eff) const
    {
        const ir::UserConstantManager& ucm = eff->getUserConstantManager();

        if (!isSystemConst(h))
        {
            return ucm.isConstantUpdated(toUserConstIndex(h));
        }
        else
        {
            if (eff->isForceUpdateOfSystemConstantsNeeded())
                return true; //first time buffers are filled

            // TODO: provide values a bit more real

            switch (toSystemConst(h))
            {
            case CmdProcSystemConst::kDT:
            {
                return true;
            }
            case CmdProcSystemConst::kElapsedTime:
            {
                return true;
            }
            case CmdProcSystemConst::kFrame:
            {
                return true;
            }
            case CmdProcSystemConst::kScreenSize:
            {
                return true;
            }
            case CmdProcSystemConst::kCaptureState:
            {
                return true;
            }
            case CmdProcSystemConst::kTileUV:
            {
                return true;
            }
            case CmdProcSystemConst::kDepthAvailable:
            {
                return true;
            }
            case CmdProcSystemConst::kHDRAvailable:
            {
                return true;
            }
            case CmdProcSystemConst::kHUDlessAvailable:
            {
                return true;
            }
            default:
                return false;
            }
        }
    }

    void MultiPassProcessor::markConstantClean(CmdProcConstHandle h, MultiPassEffect *eff)
    {
        ir::UserConstantManager& ucm = eff->getUserConstantManager();

        if (!isSystemConst(h))
        {
            ucm.markConstantClean(toUserConstIndex(h));
        }
        else
        {
            // TODO: provide mark clean functionality for any system constants that might change sporadically
        }
    }

    bool MultiPassProcessor::isDepthRequiredOnStack() const
    {
        for (size_t effIdx = 0, effIdxEnd = m_effectsStackInternal.size(); effIdx < effIdxEnd; ++effIdx)
        {
            MultiPassEffect * eff = m_effectsStackInternal[effIdx];
            if (eff->isDepthRequired())
                return true;
        }
        return false;
    }
    bool MultiPassProcessor::isHUDlessRequiredOnStack() const
    {
        for (size_t effIdx = 0, effIdxEnd = m_effectsStackInternal.size(); effIdx < effIdxEnd; ++effIdx)
        {
            MultiPassEffect * eff = m_effectsStackInternal[effIdx];
            if (eff->isHUDlessRequired())
                return true;
        }
        return false;
    }
    bool MultiPassProcessor::isHDRRequiredOnStack() const
    {
        for (size_t effIdx = 0, effIdxEnd = m_effectsStackInternal.size(); effIdx < effIdxEnd; ++effIdx)
        {
            MultiPassEffect * eff = m_effectsStackInternal[effIdx];
            if (eff->isHDRRequired())
                return true;
        }
        return false;
    }

    //returns true if effect rebuild is needed
    bool MultiPassProcessor::setInputs(
                const ir::Effect::InputData &   finalColorInput,
                const ir::Effect::InputData &   depthInput,
                const ir::Effect::InputData &   hudlessInput,
                const ir::Effect::InputData &   hdrInput,
                MultipassConfigParserError& err
                )
    {
        bool needRebuildStack = false;
        bool relinkInputsRequired = false;

        needRebuildStack |= finalColorInput.width != m_width;
        needRebuildStack |= finalColorInput.height != m_height;
        needRebuildStack |= finalColorInput.format != m_colorSourceTextureFormat;
        relinkInputsRequired |= finalColorInput.texture != m_colorSourceTexture;

        // We only care about depth inputs if any effect on the stack actually requires it
        bool isDepthRequired = isDepthRequiredOnStack();
        if (isDepthRequired)
        {
            needRebuildStack |= depthInput.width != m_depthWidth;
            needRebuildStack |= depthInput.height != m_depthHeight;
            needRebuildStack |= depthInput.format != m_depthSourceTextureFormat;
            relinkInputsRequired |= depthInput.texture != m_depthSourceTexture;
        }

        // We only care about HUDless inputs if any effect on the stack actually requires it
        bool isHUDlessRequired = isHUDlessRequiredOnStack();
        if (isHUDlessRequired)
        {
            needRebuildStack |= hudlessInput.width != m_hudlessWidth;
            needRebuildStack |= hudlessInput.height != m_hudlessHeight;
            needRebuildStack |= hudlessInput.format != m_hudlessSourceTextureFormat;
            relinkInputsRequired |= hudlessInput.texture != m_hudlessSourceTexture;
        }

        // We only care about HDR inputs if any effect on the stack actually requires it
        bool isHDRRequired = isHDRRequiredOnStack();
        if (isHDRRequired)
        {
            needRebuildStack |= hdrInput.width != m_hdrWidth;
            needRebuildStack |= hdrInput.height != m_hdrHeight;
            needRebuildStack |= hdrInput.format != m_hdrSourceTextureFormat;
            relinkInputsRequired |= hdrInput.texture != m_hdrSourceTexture;
        }

        m_width = finalColorInput.width;
        m_height = finalColorInput.height;
        m_colorSourceTextureFormat = finalColorInput.format;
        m_colorSourceTexture = finalColorInput.texture;

        m_depthWidth = depthInput.width;
        m_depthHeight = depthInput.height;
        m_depthSourceTextureFormat = depthInput.format;
        m_depthSourceTexture = depthInput.texture;

        m_hudlessWidth = hudlessInput.width;
        m_hudlessHeight = hudlessInput.height;
        m_hudlessSourceTextureFormat = hudlessInput.format;
        m_hudlessSourceTexture = hudlessInput.texture;

        m_hdrWidth = hdrInput.width;
        m_hdrHeight = hdrInput.height;
        m_hdrSourceTextureFormat = hdrInput.format;
        m_hdrSourceTexture = hdrInput.texture;

        if (relinkInputsRequired)
        {
            MultipassConfigParserError relinkErr(shadermod::MultipassConfigParserErrorEnum::eOK);
            relinkErr = relinkEffects(true);
            if (relinkErr)
            {
                needRebuildStack = true;
            }
        }

        if (needRebuildStack)
        {
            //err = rebuildEffectsInStack(0);
            //onLastEffectUpdated(!err);

            return true;
        }

        return false;
    }

    bool MultiPassProcessor::areInputsValid() const
    {
        if (m_height == undefinedSize || m_width == undefinedSize)
            return false;

        if (m_colorSourceTexture)
        {
            if (m_colorSourceTextureFormat == ir::FragmentFormat::kNUM_ENTRIES)
                return false;
        }

        if (m_depthSourceTexture)
        {
            if (m_depthSourceTextureFormat == ir::FragmentFormat::kNUM_ENTRIES)
                return false;

            if (m_depthHeight == undefinedSize || m_depthWidth == undefinedSize)
                return false;
        }

        if (m_hudlessSourceTexture)
        {
            if (m_hudlessSourceTextureFormat == ir::FragmentFormat::kNUM_ENTRIES)
                return false;

            if (m_hudlessHeight == undefinedSize || m_hudlessWidth == undefinedSize)
                return false;
        }

        if (m_hdrSourceTexture)
        {
            if (m_hdrSourceTextureFormat == ir::FragmentFormat::kNUM_ENTRIES)
                return false;

            if (m_hdrHeight == undefinedSize || m_hdrWidth == undefinedSize)
                return false;
        }

        return true;
    }

    //returns true if effect rebuild is needed
    bool MultiPassProcessor::setWidthHeight(
                unsigned int    width,
                unsigned int    height,
                unsigned int    depthWidth,
                unsigned int    depthHeight,
                unsigned int    hudlessWidth,
                unsigned int    hudlessHeight,
                unsigned int    hdrWidth,
                unsigned int    hdrHeight,
                MultipassConfigParserError& err
                )
    {
        bool needRebuildStack = false;

        needRebuildStack |= width != m_width;
        needRebuildStack |= height != m_height;
        needRebuildStack |= depthWidth != m_depthWidth;
        needRebuildStack |= depthHeight != m_depthHeight;

        if (isHUDlessRequiredOnStack())
        {
            needRebuildStack |= hudlessWidth != m_hudlessWidth;
            needRebuildStack |= hudlessHeight != m_hudlessHeight;
        }
        if (isHDRRequiredOnStack())
        {
            needRebuildStack |= hdrWidth != m_hdrWidth;
            needRebuildStack |= hdrHeight != m_hdrHeight;
        }

        m_width = width;
        m_height = height;
        m_depthWidth = depthWidth;
        m_depthHeight = depthHeight;
        m_hudlessWidth = hudlessWidth;
        m_hudlessHeight = hudlessHeight;
        m_hdrWidth = hdrWidth;
        m_hdrHeight = hdrHeight;

        if (needRebuildStack)
        {
            //err = rebuildEffectsInStack(0);
            //onLastEffectUpdated(!err);

            return true;
        }

        return false;
    }

    MultipassConfigParserError MultiPassProcessor::rebuildEffectsInStack(unsigned int startFrom, bool compareHashes)
    {
        if (!m_effectsStackInternal.size())
            return  MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);

        //we may set inputs after effects, we don't want to get errors here
        if (!areInputsValid())
            return  MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);

        assert(startFrom < m_effectsStackInternal.size());

        ir::Effect::InputData finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput;

        finalColorInput.width = m_width;
        finalColorInput.height = m_height;
        finalColorInput.format = m_colorSourceTextureFormat;
        finalColorInput.texture = m_colorSourceTexture;

        depthInput.width = m_depthWidth;
        depthInput.height = m_depthHeight;
        depthInput.format = m_depthSourceTextureFormat;
        depthInput.texture = m_depthSourceTexture;

        hudlessInput.width = m_hudlessWidth;
        hudlessInput.height = m_hudlessHeight;
        hudlessInput.format = m_hudlessSourceTextureFormat;
        hudlessInput.texture = m_hudlessSourceTexture;

        hdrInput.width = m_hdrWidth;
        hdrInput.height = m_hdrHeight;
        hdrInput.format = m_hdrSourceTextureFormat;
        hdrInput.texture = m_hdrSourceTexture;

        colorBaseInput.width = m_width;
        colorBaseInput.height = m_height;
        colorBaseInput.format = m_colorSourceTextureFormat;
        colorBaseInput.texture = m_colorSourceTexture;

        if (startFrom != 0)
        {
            MultiPassEffect* eff = m_effectsStackInternal[startFrom - 1];

            finalColorInput.width = eff->getOutputWidth();
            finalColorInput.height = eff->getOutputHeight();
            finalColorInput.format = eff->getOutFormat();
            finalColorInput.texture = eff->getOutputColorTexture();
        }

        for (auto it = m_effectsStackInternal.begin() + startFrom, end = m_effectsStackInternal.end(); it != end; ++it)
        {
            MultiPassEffect* eff = *it;

            MultipassConfigParserError err(MultipassConfigParserErrorEnum::eOK);
            MultiPassEffect* newEff = createEffectFromACEF(
                        eff->getFxToolFilepath().c_str(),
                        eff->getRootDir().c_str(),
                        eff->getTempsDir().c_str(),
                        eff->getFxFilename().c_str(),
                        finalColorInput,
                        depthInput,
                        hudlessInput,
                        hdrInput,
                        colorBaseInput,
                        err,
                        &eff->getExpectedHashSet(),
                        compareHashes
                        );

            if (err)
            {
                m_isDefunctEffectInStack = true;
                return err;
            }

            finalColorInput.width = newEff->getOutputWidth();
            finalColorInput.height = newEff->getOutputHeight();
            finalColorInput.format = newEff->getOutFormat();
            finalColorInput.texture = newEff->getOutputColorTexture();

            destroyEffect(eff);

            *it = newEff;
        }

        m_isDefunctEffectInStack = false;

        return  MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
    }

    bool MultiPassProcessor::createCopyShader()
    {
        // Create the pixel shader
        HRESULT hr = m_d3dDevice->CreatePixelShader(g_copyPS, sizeof(g_copyPS), nullptr, &m_copyPixelShader);

        if (FAILED(hr))
        {
            m_copyPixelShader = nullptr;
            return false;
        }

        // Create the sample state
        D3D11_SAMPLER_DESC sampDesc;
        ZeroMemory(&sampDesc, sizeof(sampDesc));
        sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sampDesc.MinLOD = 0;
        sampDesc.MaxLOD = D3D11_FLOAT32_MAX;

        sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        hr = m_d3dDevice->CreateSamplerState(&sampDesc, &m_samplerLinear);

        if (FAILED(hr))
        {
            m_samplerLinear = nullptr;
            return false;
        }

        m_isValid = true;

        return true;
    }

    void MultiPassProcessor::destroyCopyShader()
    {
        m_isValid = false;

        SAFE_RELEASE(m_copyPixelShader);
        SAFE_RELEASE(m_samplerLinear);
    }

    bool MultiPassProcessor::updateOutSRV()
    {
        destroyOutSRV();

        ir::FragmentFormat newOutformat = ir::FragmentFormat::kNUM_ENTRIES;
        ID3D11Texture2D* outTexture = nullptr;

        if (m_effectsStackInternal.size())
        {
            newOutformat = m_effectsStackInternal.back()->getOutFormat();
            outTexture = m_effectsStackInternal.back()->getOutputColorTexture();
        }

        if (newOutformat != ir::FragmentFormat::kNUM_ENTRIES && outTexture)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
            shaderResourceViewDesc.Format = lwanselutils::getSRVFormatDepth(lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(ir::ircolwert::formatToDXGIFormat(newOutformat))));
            shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
            shaderResourceViewDesc.Texture2D.MipLevels = 1;

            // Create the shader resource view.
            HRESULT hr = m_d3dDevice->CreateShaderResourceView(outTexture, &shaderResourceViewDesc, &m_copyInputSRV);

            if (FAILED(hr))
            {
                m_copyInputSRV = nullptr;
                return false;
            }
        }

        return true;
    }

    void MultiPassProcessor::destroyOutSRV()
    {
        if (m_copyInputSRV)
            m_copyInputSRV->Release(), m_copyInputSRV = nullptr;
    }

    bool MultiPassProcessor::copyData(ID3D11ShaderResourceView * source, ID3D11RenderTargetView * dest, float width, float height, bool skipIAVSSetup)
    {
        if (!m_immediateContext)
            return false;

        if (!isDeviceValid())
            return false;

        if (!skipIAVSSetup)
        {
            m_immediateContext->ClearState();

            m_immediateContext->IASetInputLayout(NULL);
            m_immediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            m_immediateContext->VSSetShader(m_vertexShader, nullptr, 0);
        }

        D3D11_VIEWPORT viewPort;
        viewPort.MinDepth = 0.0f;
        viewPort.MaxDepth = 1.0f;
        viewPort.TopLeftX = 0;
        viewPort.TopLeftY = 0;
        viewPort.Width = width;
        viewPort.Height = height;
        m_immediateContext->RSSetViewports(1, &viewPort);
        m_immediateContext->VSSetShader(m_vertexShader, nullptr, 0);
        m_immediateContext->PSSetShader(m_copyPixelShader, nullptr, 0);
        m_immediateContext->PSSetSamplers(0, 1, &m_samplerLinear);
        m_immediateContext->PSSetShaderResources(0, 1, &source);
        m_immediateContext->OMSetRenderTargets(1, &dest, nullptr);

        m_immediateContext->Draw(3, 0);

        return true;
    }

    bool MultiPassProcessor::processData(ID3D11RenderTargetView* dest)
    {
        assert(isDeviceValid());

        if (!m_immediateContext)
            return false;

        if (!isDeviceValid() || !areInputsValid() || isDefunctEffectOnStack())
            return false;

        m_immediateContext->ClearState();

        // Set the input layout
        m_immediateContext->IASetInputLayout(NULL);

        // Set primitive topology
        m_immediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        D3D11_VIEWPORT viewPort;
        viewPort.MinDepth = 0.0f;
        viewPort.MaxDepth = 1.0f;
        viewPort.TopLeftX = 0;
        viewPort.TopLeftY = 0;

        m_dt = m_timer.Time();
        m_elapsedTime += m_dt * 0.001f;
        m_timer.Start();

        LOG_VERBOSE("Starting post process rendering...");

        for (UINT effIdx = 0; effIdx < m_effectsStackInternal.size(); effIdx++)
        {
            MultiPassEffect* lwrrEffect = m_effectsStackInternal[effIdx];

            if (L"Sharpen.yaml" == lwrrEffect->getFxFilename())
            {
                if (m_SharpenLwbinData.m_HWSupport)
                {
                    ID3D11Texture2D* lwbinOutput = NULL;
                    if (m_effectsStackInternal.size() == effIdx + 1) // Last effect
                    {
                        ID3D11Resource * destColorTex = NULL;
                        dest->GetResource(&destColorTex);
                        lwbinOutput = static_cast<ID3D11Texture2D *>(destColorTex);

                        if (!lwrrEffect->getLowLevelEffect().m_passes.empty())
                        {
                            const CmdProcPass & lastPass = lwrrEffect->getLowLevelEffect().m_passes[lwrrEffect->getLowLevelEffect().m_passes.size() - 1];
                            viewPort.Width = (FLOAT)lastPass.m_width;
                            viewPort.Height = (FLOAT)lastPass.m_height;
                            m_immediateContext->RSSetViewports(1, &viewPort); // So that standalone UI renders correctly
                        }
                    }
                    else
                    {
                        lwbinOutput = lwrrEffect->getLowLevelEffect().GetOutputColorTex();
                    }

                    shadermod::ir::UserConstant* lwrUC = lwrrEffect->getUserConstantManager().findByName("sharpenSlider");
                    lwrUC->getValue((float*)& m_SharpenLwbinData.m_SharpenParameter, 1);

                    lwrUC = lwrrEffect->getUserConstantManager().findByName("denoiseSlider");
                    lwrUC->getValue((float*)& m_SharpenLwbinData.m_DenoiseParameter, 1);

                    if (S_OK == RunLwbin(lwrrEffect->getLowLevelEffect().GetInputColorTex(), lwbinOutput, lwrrEffect->getOutputWidth(), lwrrEffect->getOutputHeight()))
                    {
                        LOG_INFO("Ran lwbin!");
                        continue;
                    }
                    LOG_ERROR("Failed to run lwbin.");
                }
                else
                {
                    LOG_INFO("Lwbin not supported on this HW - Fallback to Sharpen HLSL shader.");
                }
            }

            m_referencedConstantsCache.clear();

            float prevVPWidth = -1.0f, prevVPHeight = -1.0f;
            for (size_t i = 0, iend = lwrrEffect->getLowLevelEffect().m_passes.size(); i < iend; ++i)
            {
                const CmdProcPass & lwrPass = lwrrEffect->getLowLevelEffect().m_passes[i];

                m_immediateContext->RSSetState(lwrPass.m_rasterizerState);
                m_immediateContext->OMSetDepthStencilState(lwrPass.m_depthStencilState, 0xFFFFFFFF);
                m_immediateContext->OMSetBlendState(lwrPass.m_alphaBlendState, NULL, 0xffffffff);

#if FORCE_CLEAR_RT_VIEW == 1
                const FLOAT black[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

                for (auto rt : lwrPass.m_renderTargets)
                    m_immediateContext->ClearRenderTargetView(rt, black);
#endif
                if (m_effectsStackInternal.size() == effIdx + 1 && lwrrEffect->getLowLevelEffect().m_passes.size() == i + 1) // Last pass of last effect
                {
                    m_immediateContext->OMSetRenderTargets((UINT)(1), &dest, nullptr);
                }
                else
                {
                    m_immediateContext->OMSetRenderTargets((UINT)lwrPass.m_renderTargets.size(), &lwrPass.m_renderTargets[0], nullptr);
                }

                if (prevVPWidth != lwrPass.m_width || prevVPHeight != lwrPass.m_height)
                {
                    viewPort.Width = (FLOAT)lwrPass.m_width;
                    viewPort.Height = (FLOAT)lwrPass.m_height;
                    m_immediateContext->RSSetViewports(1, &viewPort);
                    prevVPWidth = lwrPass.m_width;
                    prevVPHeight = lwrPass.m_height;
                }

                if (lwrPass.m_vertexShader)
                {
                    m_immediateContext->VSSetShader(lwrPass.m_vertexShader, nullptr, 0);
                }
                else
                {
                    // If no VS specified, use default vertex shader
                    m_immediateContext->VSSetShader(m_vertexShader, nullptr, 0);
                }

                m_immediateContext->PSSetShader(lwrPass.m_pixelShader, nullptr, 0);

                for (size_t j = 0, jend = lwrPass.m_samplerDescs.size(); j < jend; ++j)
                {
                    if (lwrPass.m_samplerDescs[j].slot == IR_REFLECTION_NOT_FOUND)
                        continue;

                    m_immediateContext->PSSetSamplers(lwrPass.m_samplerDescs[j].slot, 1, &lwrPass.m_samplerDescs[j].pSampler);
                }
                for (size_t j = 0, jend = lwrPass.m_shaderResourceDescs.size(); j < jend; ++j)
                {
                    if (lwrPass.m_shaderResourceDescs[j].slot == IR_REFLECTION_NOT_FOUND)
                        continue;

                    if (lwrPass.m_shaderResourceDescs[j].pResource)
                    {
                        m_immediateContext->GenerateMips(lwrPass.m_shaderResourceDescs[j].pResource);
                        m_immediateContext->PSSetShaderResources(lwrPass.m_shaderResourceDescs[j].slot, 1, &lwrPass.m_shaderResourceDescs[j].pResource);
                    }
                }

                for (size_t j = 0, jend = lwrPass.m_constantBufPSDescs.size(); j < jend; ++j)
                {
                    const CmdProcConstantBufDesc & lwrConstBuf = lwrPass.m_constantBufPSDescs[j];

                    if (lwrConstBuf.m_slot == IR_REFLECTION_NOT_FOUND)
                        continue;

                    bool needUpdate = false;

                    for (const auto& c : lwrConstBuf.m_constants)
                    {
                        if (isConstantDirty(c.constHandle, lwrrEffect))
                        {
                            needUpdate = true;
                            break;
                        }
                    }

                    if (needUpdate)
                    {
                        D3D11_MAPPED_SUBRESOURCE subResource;
                        m_immediateContext->Map(lwrConstBuf.m_pBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);

                        // Do the constant setup here
                        for (size_t k = 0, kend = lwrConstBuf.m_constants.size(); k < kend; ++k)
                        {
                            const CmdProcConstantDesc & lwrConst = lwrConstBuf.m_constants[k];
                            void * offsetPtr = (void *)((float *)subResource.pData + lwrConst.offsetInComponents);
                            bool b = writeConstantValue(lwrConst.constHandle, lwrrEffect, offsetPtr,
                                getConstantDataDimensions(lwrConst.constHandle, lwrrEffect)*getCmdProcConstDataElementTypeSize(getConstantDataType(lwrConst.constHandle, lwrrEffect)));

                            assert(b && "can't write constant value!");

                            m_referencedConstantsCache.push_back(lwrConst.constHandle);
                        }

                        m_immediateContext->Unmap(lwrConstBuf.m_pBuffer, 0);
                    }

                    m_immediateContext->PSSetConstantBuffers(lwrConstBuf.m_slot, 1, &lwrConstBuf.m_pBuffer);
                }
                for (size_t j = 0, jend = lwrPass.m_constantBufVSDescs.size(); j < jend; ++j)
                {
                    const CmdProcConstantBufDesc & lwrConstBuf = lwrPass.m_constantBufVSDescs[j];

                    if (lwrConstBuf.m_slot == IR_REFLECTION_NOT_FOUND)
                        continue;

                    bool needUpdate = false;

                    for (const auto& c : lwrConstBuf.m_constants)
                    {
                        if (isConstantDirty(c.constHandle, lwrrEffect))
                        {
                            needUpdate = true;
                            break;
                        }
                    }

                    if (needUpdate)
                    {
                        D3D11_MAPPED_SUBRESOURCE subResource;
                        m_immediateContext->Map(lwrConstBuf.m_pBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);

                        // Do the constant setup here
                        for (size_t k = 0, kend = lwrConstBuf.m_constants.size(); k < kend; ++k)
                        {
                            const CmdProcConstantDesc & lwrConst = lwrConstBuf.m_constants[k];
                            void * offsetPtr = (void *)((float *)subResource.pData + lwrConst.offsetInComponents);
                            bool b = writeConstantValue(lwrConst.constHandle, lwrrEffect, offsetPtr,
                                getConstantDataDimensions(lwrConst.constHandle, lwrrEffect)*getCmdProcConstDataElementTypeSize(getConstantDataType(lwrConst.constHandle, lwrrEffect)));

                            assert(b && "can't write constant value!");

                            m_referencedConstantsCache.push_back(lwrConst.constHandle);
                        }

                        m_immediateContext->Unmap(lwrConstBuf.m_pBuffer, 0);
                    }

                    m_immediateContext->VSSetConstantBuffers(lwrConstBuf.m_slot, 1, &lwrConstBuf.m_pBuffer);
                }

                // TODO: DBG
#if 1
                m_immediateContext->Draw(3, 0);
#else
                const FLOAT blue[4] = { 0.0f, 0.2f, 0.4f, 1.0f };
                m_immediateContext->ClearRenderTargetView(lwrPass.m_renderTargets[0], blue);
#endif

#if (CLEANUP_STATES == 1)
                ID3D11ShaderResourceView* pSRVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                ID3D11RenderTargetView * pRTVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                ID3D11DepthStencilView * pDSV = NULL;
                m_immediateContext->OMSetRenderTargets(8, pRTVs, pDSV);
                m_immediateContext->PSSetShaderResources(0, 16, pSRVs);
#endif
            }//passes

            for (auto c : m_referencedConstantsCache)
                markConstantClean(c, lwrrEffect);

            m_referencedConstantsCache.clear();

            lwrrEffect->markSystemConstantsUpdatedOnce();
        } //effects

#if (CLEANUP_STATES == 1)
        ID3D11ShaderResourceView* pSRVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        ID3D11RenderTargetView * pRTVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        ID3D11DepthStencilView * pDSV = NULL;
        m_immediateContext->OMSetRenderTargets(8, pRTVs, pDSV);
        m_immediateContext->PSSetShaderResources(0, 16, pSRVs);
#endif

        ++m_processingFrameNum;

        return true;
    }

#define SHARPEN_LWBIN_INPUT_SLOTS 6
#define NGX_DLISP_COMPOSITE_LWBIN_CTA_X 128
#define NGX_DLISP_COMPOSITE_LWBIN_CTA_Y 1
#define NGX_DLISP_COMPOSITE_LWBIN_CTA_Z 1

#define NGX_DLISP_COMPOSITE_LWBIN_TILE_WIDTH 32
#define NGX_DLISP_COMPOSITE_LWBIN_TILE_HEIGHT 32

    void MultiPassProcessor::CheckLwbinSupportOnHW(ID3D11Device *InDevice)
    {
        LW_GPU_ARCHITECTURE_ID gpuArchId;
        if (!GetGPUArch(InDevice, gpuArchId))
        {
            return;
        }

        m_SharpenLwbinData.m_HWSupport = true;

        switch (gpuArchId)
        {
        case LW_GPU_ARCHITECTURE_GK100:
        case LW_GPU_ARCHITECTURE_GK110:
        case LW_GPU_ARCHITECTURE_GK200:
            m_SharpenLwbinData.m_Kernel_LDR = sharpen_sm30_ldr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_LDR = sharpen_sm30_ldr_lwbin_len;
            m_SharpenLwbinData.m_Kernel_HDR = sharpen_sm30_hdr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_HDR = sharpen_sm30_hdr_lwbin_len;
            m_SharpenLwbinData.m_HWSupport = false;
            LOG_INFO("LW_GPU_ARCHITECTURE_GK");
            break;

        case LW_GPU_ARCHITECTURE_GM000:
        case LW_GPU_ARCHITECTURE_GM200:
            m_SharpenLwbinData.m_Kernel_LDR = sharpen_sm50_ldr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_LDR = sharpen_sm50_ldr_lwbin_len;
            m_SharpenLwbinData.m_Kernel_HDR = sharpen_sm50_hdr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_HDR = sharpen_sm50_hdr_lwbin_len;
            m_SharpenLwbinData.m_HWSupport = false;
            LOG_INFO("LW_GPU_ARCHITECTURE_GM");
            break;

        case LW_GPU_ARCHITECTURE_GP100:
            m_SharpenLwbinData.m_Kernel_LDR = sharpen_sm60_ldr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_LDR = sharpen_sm60_ldr_lwbin_len;
            m_SharpenLwbinData.m_Kernel_HDR = sharpen_sm60_hdr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_HDR = sharpen_sm60_hdr_lwbin_len;
            m_SharpenLwbinData.m_HWSupport = false;
            LOG_INFO("LW_GPU_ARCHITECTURE_GP");
            break;

        case LW_GPU_ARCHITECTURE_TU100:
            m_SharpenLwbinData.m_Kernel_LDR = sharpen_sm75_ldr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_LDR = sharpen_sm75_ldr_lwbin_len;
            m_SharpenLwbinData.m_Kernel_HDR = sharpen_sm75_hdr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_HDR = sharpen_sm75_hdr_lwbin_len;
            LOG_INFO("LW_GPU_ARCHITECTURE_TU");
            break;

        case LW_GPU_ARCHITECTURE_GA100:
            m_SharpenLwbinData.m_Kernel_LDR = sharpen_sm86_ldr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_LDR = sharpen_sm86_ldr_lwbin_len;
            m_SharpenLwbinData.m_Kernel_HDR = sharpen_sm86_hdr_lwbin;
            m_SharpenLwbinData.m_KernelSizeInBytes_HDR = sharpen_sm86_hdr_lwbin_len;
            LOG_INFO("LW_GPU_ARCHITECTURE_GA");
            break;

        default:
            m_SharpenLwbinData.m_HWSupport = false;
            LOG_INFO("LW_GPU_ARCHITECTURE - Unsupported!");
            break;
        }

        m_HWSupportsLwbin = m_SharpenLwbinData.m_HWSupport;
    }

    HRESULT MultiPassProcessor::InitLwbin()
    {
        if (m_LwbinAPI == nullptr)
        {
            LWAPI_CHECK(GetNGXLwbinD3D11()->Init(m_d3dDevice), "Lwbin API failed to init.");
            m_LwbinAPI = GetNGXLwbinD3D11();
            LWAPI_CHECK(
                m_LwbinAPI->CreateKernel(m_SharpenLwbinData.m_Kernel_LDR, m_SharpenLwbinData.m_KernelSizeInBytes_LDR, "DeepISP_Sharpen_NativeRGBA", SHARPEN_LWBIN_INPUT_SLOTS, m_SharpenKernel_LDR, NGX_DLISP_COMPOSITE_LWBIN_CTA_X, NGX_DLISP_COMPOSITE_LWBIN_CTA_Y, NGX_DLISP_COMPOSITE_LWBIN_CTA_Z),
                "Failed to create m_SharpenKernel_LDR.");
            LWAPI_CHECK(
                m_LwbinAPI->CreateKernel(m_SharpenLwbinData.m_Kernel_HDR, m_SharpenLwbinData.m_KernelSizeInBytes_HDR, "DeepISP_Sharpen_NativeRGBA", SHARPEN_LWBIN_INPUT_SLOTS, m_SharpenKernel_HDR, NGX_DLISP_COMPOSITE_LWBIN_CTA_X, NGX_DLISP_COMPOSITE_LWBIN_CTA_Y, NGX_DLISP_COMPOSITE_LWBIN_CTA_Z),
                "Failed to create m_SharpenKernel_HDR.");
        }
        LOG_DEBUG("Lwbin API Init Successful!");
        return S_OK;
    }

    HRESULT MultiPassProcessor::ShutdownLwbin()
    {
        if (m_LwbinAPI)
        {
            LWAPI_CHECK(m_LwbinAPI->DestroyKernel(m_SharpenKernel_LDR), "Failed to destroy m_SharpenKernel_LDR.");
            LWAPI_CHECK(m_LwbinAPI->DestroyKernel(m_SharpenKernel_HDR), "Failed to destroy m_SharpenKernel_HDR.");
            LWAPI_CHECK(m_LwbinAPI->Shutdown(), "Failed to shutdown Lwbin API.");
        }
        return S_OK;
    }

    HRESULT MultiPassProcessor::ColwertToUAVSupportedFormat(DXGI_FORMAT& format)
    {
        UAVSupport uavs = GetUAVFormatSupport(format);
        if (uavs.UAVSupported()) return S_OK;

        DXGI_FORMAT formatNew = format;
        switch (format)
        {
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: formatNew = DXGI_FORMAT_R32G32B32A32_FLOAT; break;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: formatNew = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
        case DXGI_FORMAT_R16G16B16A16_UNORM: formatNew = DXGI_FORMAT_R16G16B16A16_UINT; break;
        case DXGI_FORMAT_R16G16B16A16_SNORM: formatNew = DXGI_FORMAT_R16G16B16A16_SINT; break;
        case DXGI_FORMAT_R32G32_TYPELESS: formatNew = DXGI_FORMAT_R32G32_FLOAT; break;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: formatNew = DXGI_FORMAT_R10G10B10A2_UNORM; break;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: formatNew = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: formatNew = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case DXGI_FORMAT_R8G8B8A8_SNORM: formatNew = DXGI_FORMAT_R8G8B8A8_SINT; break;
        case DXGI_FORMAT_R16G16_TYPELESS: formatNew = DXGI_FORMAT_R16G16_FLOAT; break;
        case DXGI_FORMAT_R32_TYPELESS: formatNew = DXGI_FORMAT_R32_FLOAT; break;
        case DXGI_FORMAT_R8G8_TYPELESS: formatNew = DXGI_FORMAT_R8G8_UNORM; break;
        case DXGI_FORMAT_R8G8_SINT: formatNew = DXGI_FORMAT_R8G8_SNORM; break;
        case DXGI_FORMAT_R16_TYPELESS: formatNew = DXGI_FORMAT_R16_FLOAT; break;
        case DXGI_FORMAT_R16_UNORM: formatNew = DXGI_FORMAT_R16_UINT; break;
        case DXGI_FORMAT_R16_SNORM: formatNew = DXGI_FORMAT_R16_SINT; break;
        case DXGI_FORMAT_R8_TYPELESS: formatNew = DXGI_FORMAT_R8_UNORM; break;
        case DXGI_FORMAT_R8_SNORM: formatNew = DXGI_FORMAT_R8_UNORM; break;
        case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: formatNew = DXGI_FORMAT_R10G10B10A2_UNORM; break;
        }

        if (formatNew == format) return E_FAIL;

        UAVSupport uavsNew = GetUAVFormatSupport(formatNew);
        if (uavsNew.UAVSupported())
        {
            format = formatNew;
            return S_OK;
        }
        return E_FAIL;
    }

    UAVSupport MultiPassProcessor::GetUAVFormatSupport(DXGI_FORMAT format)
    {
        if (m_uavFormatSupport.find(format) != m_uavFormatSupport.end())
        {
            return m_uavFormatSupport[format];
        }

        UAVSupport support;

        if (!m_d3dDevice)
        {
            LOG_ERROR("No d3dDevice available in GetUAVFormatSupport");
            return support;
        }

        HRESULT hr;
        D3D11_FEATURE_DATA_FORMAT_SUPPORT FormatSupport = { format, (D3D11_FORMAT_SUPPORT)(0) };
        hr = m_d3dDevice->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT, &FormatSupport, sizeof(FormatSupport));
        if (SUCCEEDED(hr))
        {
            support.D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW = (FormatSupport.OutFormatSupport & D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW) != 0;
        }
        else
        {
            LOG_VERBOSE("Failed to get D3D11_FEATURE_FORMAT_SUPPORT for format %s", DxgiFormat_cstr(format));
        }

        D3D11_FEATURE_DATA_FORMAT_SUPPORT2 FormatSupport2 = { format, (D3D11_FORMAT_SUPPORT2)(0) };
        hr = m_d3dDevice->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &FormatSupport2, sizeof(FormatSupport2));
        if (SUCCEEDED(hr))
        {
            support.D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD = (FormatSupport2.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD) != 0;
            support.D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE = (FormatSupport2.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;
        }
        else
        {
            LOG_VERBOSE("Failed to get D3D11_FEATURE_FORMAT_SUPPORT2 for format %s", DxgiFormat_cstr(format));
        }
        return support;
    }

    void MultiPassProcessor::LogUAVFormatSupport(DXGI_FORMAT format, const UAVSupport& support)
    {
        LOG_VERBOSE("UAV Format Support for DXGI_FORMAT: %s", DxgiFormat_cstr(format));
        LOG_VERBOSE("  D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW: %s", support.D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW ? "TRUE" : "FALSE");
        LOG_VERBOSE("  D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD: %s", support.D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD ? "TRUE" : "FALSE");
        LOG_VERBOSE("  D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE: %s", support.D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE ? "TRUE" : "FALSE");
    }

    HRESULT MultiPassProcessor::RunLwbin(ID3D11Resource *Input, ID3D11Resource *Output, UINT Width, UINT Height)
    {
        LOG_DEBUG("RunLwbin(Input = 0x%llx, Output = 0x%llx, Width = %d, Height = %d)", Input, Output, Width, Height);

        if (!Input || !Output)
        {
            return E_ILWALIDARG;
        }

        if (!m_LwbinAPI)
        {
            if (InitLwbin() != S_OK)
            {
                m_HWSupportsLwbin = m_SharpenLwbinData.m_HWSupport = false;
                return E_FAIL;
            }
        }

        // Sharpen final output
        if (!m_SharpenParams)
        {
            m_SharpenParams = new NGXLwbinParameters();
            m_SharpenParams->Init(SHARPEN_LWBIN_INPUT_SLOTS);
        }

        const float kSharpnessMin = -1.0f / 14.0f;
        const float kSharpnessMax = -1.0f / 6.5f;
        float KernelSharpness = kSharpnessMin + (kSharpnessMax - kSharpnessMin) * min(max(m_SharpenLwbinData.m_SharpenParameter, 0.0f), 1.0f);

        const float kDenoiseMin = 0.001f;
        const float kDenoiseMax = 0.1f;
        float KernelDenoise = 1.0f / (kDenoiseMin + (kDenoiseMax - kDenoiseMin) * min(max(m_SharpenLwbinData.m_DenoiseParameter, 0.0f), 1.0f));

        float IlwWidth = 1.0f / Width;
        float IntHeight = 1.0f / Height;
        float Sharpness = KernelSharpness;
        float Denoise = KernelDenoise;

        NGXLwbin_Resource_Description DescOutput = {};
        LwAPI_Status DescStatus = m_LwbinAPI->GetResourceDescription(Output, DescOutput);
        if (DescStatus == LWAPI_OK)
        {
            DXGI_FORMAT OutputColwertedFormat = lwanselutils::colwertFromTypelessIfNeeded((DXGI_FORMAT)(DescOutput.Format));
            UAVSupport uavSupport = GetUAVFormatSupport(OutputColwertedFormat);
            if (!uavSupport.D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW
                || !uavSupport.D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD
                || !uavSupport.D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE)
            {
                LOG_WARN("Did not find full UAV Support for %s", DxgiFormat_cstr(DescOutput.Format));
                LogUAVFormatSupport((DXGI_FORMAT)(DescOutput.Format), uavSupport);
            }
        }
        else
        {
            LOG_WARN("Failed to get output buffer description.");
            return E_FAIL;
        }

        bool isInputHDR = false;
        NGXLwbin_Resource_Description DescInput = {};
        DescStatus = m_LwbinAPI->GetResourceDescription(Input, DescInput);
        if (DescStatus == LWAPI_OK)
        {
            DXGI_FORMAT InputColwertedFormat = lwanselutils::colwertFromTypelessIfNeeded((DXGI_FORMAT)(DescInput.Format));
            isInputHDR = isHdrFormatSupported(InputColwertedFormat);
            LOG_VERBOSE("Input buffer's %s is %sHDR", DxgiFormat_cstr(DescInput.Format), isInputHDR ? "" : "*NOT* ");
        }
        else
        {
            LOG_WARN("Failed to get lwbin Input Buffer resource description");
        }
        D3D11_BIND_UNORDERED_ACCESS;
        LogResourceDescription("Lwbin Input Texture", DescInput);
        LogResourceDescription("Lwbin Output Surface", DescOutput);
        LOG_DEBUG("Lwbin (IlwWidth,IntHeight): (%f,%f)", IlwWidth, IntHeight);
        LOG_DEBUG("Lwbin Sharpness: %f", Sharpness);
        LOG_DEBUG("Lwbin Denoise: %f", Denoise);

        if (isInputHDR)
        {
            LWAPI_CHECK(m_LwbinAPI->BindKernel(m_SharpenKernel_HDR), "Failed to bind m_SharpenKernel_HDR");
        }
        else
        {
            LWAPI_CHECK(m_LwbinAPI->BindKernel(m_SharpenKernel_LDR), "Failed to bind m_SharpenKernel_LDR");
        }

        LWAPI_CHECK(m_LwbinAPI->SetInputSurface(0, Output, m_SharpenParams), "Failed to set input 0");
        LWAPI_CHECK(m_LwbinAPI->SetInputTexture(1, Input, m_SharpenParams), "Failed to set input 1");
        LWAPI_CHECK(m_LwbinAPI->SetInput(2, IlwWidth, m_SharpenParams), "Failed to set input 2");
        LWAPI_CHECK(m_LwbinAPI->SetInput(3, IntHeight, m_SharpenParams), "Failed to set input 3");
        LWAPI_CHECK(m_LwbinAPI->SetInput(4, Sharpness, m_SharpenParams), "Failed to set input 4");
        LWAPI_CHECK(m_LwbinAPI->SetInput(5, Denoise, m_SharpenParams), "Failed to set input 5");
        LWAPI_CHECK(m_LwbinAPI->Dispatch(m_SharpenParams, m_immediateContext, (Width + NGX_DLISP_COMPOSITE_LWBIN_TILE_WIDTH - 1) / NGX_DLISP_COMPOSITE_LWBIN_TILE_WIDTH,
            (Height + NGX_DLISP_COMPOSITE_LWBIN_TILE_HEIGHT - 1) / NGX_DLISP_COMPOSITE_LWBIN_TILE_HEIGHT),
            "Failed to dispatch");

        return S_OK;
    }
}
