#pragma once

namespace shadermod
{
namespace ir
{

    class Sampler
    {
    public:

        Sampler::Sampler(AddressType addrU, AddressType addrV, FilterType filterMin, FilterType filterMag, FilterType filterMip) :
            m_addrU(addrU),
            m_addrV(addrV),
            m_filterMin(filterMin),
            m_filterMag(filterMag),
            m_filterMip(filterMip)
        {
        }

        AddressType         m_addrU;
        AddressType         m_addrV;
        AddressType         m_addrW = AddressType::kClamp;

        FilterType          m_filterMin;
        FilterType          m_filterMag;
        FilterType          m_filterMip = FilterType::kPoint;

        // GAPI-specific
        ID3D11SamplerState *    m_D3DSampler = nullptr;

    protected:
    };

}
}
