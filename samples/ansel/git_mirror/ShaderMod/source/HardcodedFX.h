#pragma once
#include <unordered_map>

std::unordered_map<std::wstring, std::wstring>& GetHardcodedFXFiles();
bool IsEffectHardcoded(const std::wstring& effectNameNoExt);
HRESULT WriteHardcodedBinaryFile(const std::wstring& effectNameNoExt, const std::wstring& fullFilePath);
HRESULT AddHardcodedEffectsAndAlphabetize(std::vector<std::wstring>& effectFilesList, std::vector<std::wstring>& effectRootFoldersLis, 
                                        const std::wstring& effectInstallationFolderPath);
