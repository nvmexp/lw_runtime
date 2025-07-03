#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_set>
#include "Log.h"
#include "ir\FileHelpers.h"
#include "HardcodedFX.h"
#include "filterspecific\HardcodedFiles.h"

std::unordered_map<std::wstring, std::wstring>& GetHardcodedFXFiles_ForceNoCheck()
{
    static std::unordered_map<std::wstring, std::wstring> hardcodedFXFiles;
    return hardcodedFXFiles;
}

bool IsEffectHardcoded(const std::wstring& effectNameNoExt)
{
    std::unordered_map<std::wstring, std::wstring>& hardcodedFXFiles = GetHardcodedFXFiles();
    return (hardcodedFXFiles.find(effectNameNoExt) != hardcodedFXFiles.end());
}

// This can either write a hardcoded binary file to disc, or populate the list of hardcoded effects.
HRESULT processHardcodedBinaryFileInfo(const std::wstring& effectNameNoExt, const std::wstring& fullFilePath, bool populateTheListOfHardcodedEffects)
{
    // Only check to populate the list of hardcoded effects once.
    std::unordered_map<std::wstring, std::wstring>* pHardcodedFXFiles = NULL;
    static bool isListOfHardcodedEffectsPopulated = false;
    if (populateTheListOfHardcodedEffects)
    {
        if (!isListOfHardcodedEffectsPopulated)
        {
            pHardcodedFXFiles = &GetHardcodedFXFiles_ForceNoCheck();
            pHardcodedFXFiles->clear();
            isListOfHardcodedEffectsPopulated = true;
        }
        else
        {
            return S_OK;
        }
    }

    // Only write the file once per app run. This also ensures that the effect file will be updated if a new version gets hardcoded.
    static std::unordered_set<std::wstring> hardcodedFilesWritten;
    if (hardcodedFilesWritten.find(fullFilePath) != hardcodedFilesWritten.end())
    {
        return S_OK;
    }
    hardcodedFilesWritten.insert(fullFilePath);

    std::wstring directory, fileName;
    shadermod::ir::filehelpers::SplitPathIntoDirectoryAndFileName(fullFilePath, directory, fileName);
    shadermod::ir::filehelpers::createDirectoryRelwrsively(directory.c_str());

    return CheckForFilesOrWriteFiles(effectNameNoExt, fullFilePath, directory, fileName, pHardcodedFXFiles);
}

HRESULT WriteHardcodedBinaryFile(const std::wstring& effectNameNoExt, const std::wstring& fullFilePath)
{
    return processHardcodedBinaryFileInfo(effectNameNoExt, fullFilePath, false);
}

std::unordered_map<std::wstring, std::wstring>& GetHardcodedFXFiles()
{
    std::unordered_map<std::wstring, std::wstring>& hardcodedFXFiles = GetHardcodedFXFiles_ForceNoCheck();
    processHardcodedBinaryFileInfo(L"", L"", true); // Checks to populate the list of hardcoded effects.
    return hardcodedFXFiles;
}

HRESULT AddHardcodedEffectsAndAlphabetize(std::vector<std::wstring>& effectFilesList, std::vector<std::wstring>& effectRootFoldersLis, const std::wstring& effectInstallationFolderPath)
{
    // Add the hardcoded effects to the list
    std::unordered_map<std::wstring, std::wstring>& hardcodedFXFiles = GetHardcodedFXFiles();
    for (auto hardcodedItr = hardcodedFXFiles.begin(); hardcodedItr != hardcodedFXFiles.end(); hardcodedItr++)
    {
        if (hardcodedItr->second != L"") // Do not add intermediate sub-filters
        {
            effectFilesList.push_back(hardcodedItr->first + L"." + hardcodedItr->second);
            effectRootFoldersLis.push_back(effectInstallationFolderPath);
        }
    }

    // Create the alphabetized list
    std::map<std::wstring, std::wstring> alphabetizedEffectToRootMap;
    for (UINT i = 0; i < effectFilesList.size(); i++)
    {
        alphabetizedEffectToRootMap[effectFilesList[i]] = effectRootFoldersLis[i];
    }

    // Reset the input vectors to the alphabetized order
    effectFilesList.clear();
    effectRootFoldersLis.clear();
    for (auto mapItr = alphabetizedEffectToRootMap.begin(); mapItr != alphabetizedEffectToRootMap.end(); mapItr++)
    {
        effectFilesList.push_back(mapItr->first);
        effectRootFoldersLis.push_back(mapItr->second);
    }

    return S_OK;
}
