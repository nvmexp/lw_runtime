#pragma once

#include "Hash.h"

#include <vector>

namespace shadermod
{
    class MultiPassEffect;
}

struct EffectsInfo
{
    // List of effects (built from the list of YAML files in certain directory)
    std::vector<std::wstring>   m_effectRootFoldersList;    // Each effect's full path
    std::vector<std::wstring>   m_effectFilesList;          // Raw filenames
    std::vector<std::wstring>   m_effectFilterIds;          // Filter IDs (full filepath as of now)
    // TODO: NOT USED - remove instances from the code
    int  m_selectedEffect;

    enum class BufferToCheck : uint32_t
    {
        kDepth =        1,
        kHDR =          2,
        kHUDless =      4,

        kNONE = (uint32_t)0,
        kALL = (uint32_t)-1
    };

    // List of effects on the stack as client sees them (with all the empty/none slots)
    std::vector<int>    m_effectSelected;                           // Index of effect selected for a all the slots (m_effectFilesList array can be indexed using this)
    std::vector<bool>   m_effectRebuildRequired;                    // Whether an effect requires rebuild or not (e.g. if filterId was changed for a given slot)
    std::vector<bool>   m_bufferCheckRequired;                      // If effect requires special buffers, and a check is required to see if buffers are available
    std::vector<uint32_t> m_bufferCheckMessages;                    // Whether the effect check should produice message next time the buffer is unavailable
    std::vector<shadermod::MultiPassEffect *> m_effectsStack;       // Sparse array of effect pointers (colwenience rather than necessity) - has nullptrs where effect is empty/none
    std::vector<int>    m_effectsStackMapping;                      // Indices into dense effectsStack array in the effects framework
    
    std::vector<Hash::Data> m_effectsHashedName;                    // Effect name hashed cached
};
