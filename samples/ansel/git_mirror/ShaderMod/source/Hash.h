#pragma once

#include "sha2.hpp"

#include <array>
#include <stdint.h>
#include <string>

namespace Hash
{
    using Data = std::array<uint8_t, SHA256_DIGEST_SIZE>;

    enum class Type
    {
        Acef,
        Resource,
        Shader
    };

    class Effects
    {
    public:
        Effects() {}

        Effects(const Data& acef,
            const Data& res,
            const Data& shader,
            const std::string& effName)
            : m_acef(acef)
            , m_res(res)
            , m_shader(shader)
            , m_effName(effName) {}

        const Data& GetHash(Type type) const;

        void UpdateHash(sha256_ctx* ctx, Type type);

        const std::string GetEffName() const;

        // For each hash (acef/res/shader), add the piecewise values with those of addHash
        void Add(const Effects& addHash);

        // Needs to be defined for set<> operations to work
        bool operator<(const Effects& rhs) const;

    private:
        std::string m_effName;
        Data m_acef;
        Data m_res;
        Data m_shader;
    };

    const std::string FormatHash(const Hash::Data &hash);
    const Hash::Data UnshiftHash(const Hash::Data &hash);

    static constexpr Hash::Data s_emptyShiftedData()
    {
        Hash::Data emptyShifted = {};
        for (uint8_t i = 0; i < emptyShifted.size(); i++)
        {
            emptyShifted[i] = i;
        }
        return emptyShifted;
    };

    static const Hash::Effects s_emptyShiftedEffects =
    {
        s_emptyShiftedData(), s_emptyShiftedData(), s_emptyShiftedData(), ""
    };
}
