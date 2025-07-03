#include "hash.h"

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <sstream>
#include <tuple>

const Hash::Data& Hash::Effects::GetHash(Type type) const
{
    switch (type)
    {
        case Type::Acef:
            return m_acef;

        case Type::Resource:
            return m_res;

        case Type::Shader:
            return m_shader;

        default:
            assert(!"Unknown Hash::Type!");
    }

    static const Data empty = { 0 };
    return empty;
}

void Hash::Effects::UpdateHash(sha256_ctx* ctx, Type type)
{
    Data* pData = nullptr;
    switch (type)
    {
        case Type::Acef:
            pData = &m_acef;
            break;

        case Type::Resource:
            pData = &m_res;
            break;

        case Type::Shader:
            pData = &m_shader;
            break;

        default:
            assert(!"Unknown Hash::Type!");
            return;
    }

    sha256_final(ctx, pData->data());
}

// For each hash (acef/res/shader), add the piecewise values with those of addHash
void Hash::Effects::Add(const Effects& addHash)
{
    std::transform(m_acef.begin(),   m_acef.end(),   addHash.GetHash(Hash::Type::Acef).begin(),     m_acef.begin(),   std::plus<uint8_t>());
    std::transform(m_res.begin(),    m_res.end(),    addHash.GetHash(Hash::Type::Resource).begin(), m_res.begin(),    std::plus<uint8_t>());
    std::transform(m_shader.begin(), m_shader.end(), addHash.GetHash(Hash::Type::Shader).begin(),   m_shader.begin(), std::plus<uint8_t>());
}

bool Hash::Effects::operator<(const Effects& rhs) const
{
    // This will start with a comparison of Acef first, and if this >= rhs, move onto comparing Resource,
    // the repeat the logic once more for Shader
    return std::tie(this->GetHash(Type::Acef),  this->GetHash(Type::Resource),      this->GetHash(Type::Shader)) <
           std::tie(rhs.GetHash(Type::Acef),    rhs.GetHash(Type::Resource),        rhs.GetHash(Type::Shader));
}

const std::string Hash::Effects::GetEffName() const
{
    return m_effName;
}

// Returns a string of the format "{0xaa, 0xbb, 0xcc, ...}"
const std::string Hash::FormatHash(const Hash::Data &hash)
{
    std::stringstream byteStream;
    byteStream << std::hex << std::setfill('0');
    byteStream << "{";
    for (uint8_t i = 0; i < SHA256_DIGEST_SIZE; i++)
    {
        byteStream << "0x" << std::hex << std::setw(2) << static_cast<int>(hash[i]);
        if (i < SHA256_DIGEST_SIZE - 1)
        {
            byteStream << ", ";
        }
        else
        {
            byteStream << "}";
        }
    }
    return byteStream.str();
}

// Returns an unshifted version of the given hash
const Hash::Data Hash::UnshiftHash(const Hash::Data &hash)
{
    Hash::Data outHash = {};
    for (size_t hashByte = 0; hashByte < SHA256_DIGEST_SIZE; ++hashByte)
    {
        outHash[hashByte] = hash[hashByte] - (uint8_t)hashByte;
    }
    return outHash;
}
