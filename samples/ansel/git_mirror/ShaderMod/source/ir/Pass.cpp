#include <stdio.h>
#include <vector>
#include <stdint.h>

#include <d3d11_1.h>
#include <D3D11Shader.h>

#include "ir/Defines.h"
#include "ir/TypeEnums.h"
#include "ir/SpecializedPool.h"
#include "ir/Sampler.h"
#include "ir/Constant.h"
#include "ir/DataSource.h"
#include "ir/Texture.h"
#include "ir/ShaderHelpers.h"
#include "ir/VertexShader.h"
#include "ir/PixelShader.h"
#include "ir/PipelineStates.h"
#include "ir/Pass.h"

namespace shadermod
{
namespace ir
{

    DataSource * Pass::addDataSource(DataSource * dataSrc, int slot, int mrtChannel)
    {
        m_dataSources.push_back(dataSrc);
        m_mrtChannelSrc.push_back(mrtChannel);
        m_slotSrc.push_back(slot);
        m_nameSrc.push_back(nullptr);
        return dataSrc;
    }

    DataSource * Pass::addDataSource(DataSource * dataSrc, const char * name, int mrtChannel)
    {
        m_dataSources.push_back(dataSrc);
        m_mrtChannelSrc.push_back(mrtChannel);
        m_slotSrc.push_back(BindByName);

        ResourceName * resName = m_namesPool->getElement();
        new (resName) ResourceName();

        sprintf_s(resName->name, IR_RESOURCENAME_MAX*sizeof(char), "%s", name);

        m_nameSrc.push_back(resName);
        return dataSrc;
    }

    Texture * Pass::addDataOut(Texture * dataOut, int mrtChannel)
    {
        dataOut->m_needsRTV = true;

        // TODO: maybe add presort here
        m_dataOut.push_back(dataOut);
        m_mrtChannelOut.push_back(mrtChannel);
        return dataOut;
    }

    Sampler * Pass::addSampler(Sampler * sampler, int slot)
    {
        m_samplers.push_back(sampler);
        m_samplerSlots.push_back(slot);
        m_samplerNames.push_back(nullptr);
        return sampler;
    }

    Sampler * Pass::addSampler(Sampler * sampler, const char * name)
    {
        m_samplers.push_back(sampler);
        m_samplerSlots.push_back(BindByName);

        ResourceName * resName = m_namesPool->getElement();
        new (resName) ResourceName();

        sprintf_s(resName->name, IR_RESOURCENAME_MAX*sizeof(char), "%s", name);

        m_samplerNames.push_back(resName);

        return sampler;
    }

    ConstantBuf * Pass::addConstantBufferVS(ConstantBuf * constantBuf, int slot)
    {
        m_constBufsVS.push_back(constantBuf);
        m_constBufVSSlots.push_back(slot);
        m_constBufVSNames.push_back(nullptr);

        return constantBuf;
    }
    ConstantBuf * Pass::addConstantBufferVS(ConstantBuf * constantBuf, const char * name)
    {
        m_constBufsVS.push_back(constantBuf);
        m_constBufVSSlots.push_back(BindByName);

        ResourceName * resName = m_namesPool->getElement();
        new (resName) ResourceName();

        sprintf_s(resName->name, IR_RESOURCENAME_MAX*sizeof(char), "%s", name);

        m_constBufVSNames.push_back(resName);

        return constantBuf;
    }

    ConstantBuf * Pass::addConstantBufferPS(ConstantBuf * constantBuf, int slot)
    {
        m_constBufsPS.push_back(constantBuf);
        m_constBufPSSlots.push_back(slot);
        m_constBufPSNames.push_back(nullptr);

        return constantBuf;
    }
    ConstantBuf * Pass::addConstantBufferPS(ConstantBuf * constantBuf, const char * name)
    {
        m_constBufsPS.push_back(constantBuf);
        m_constBufPSSlots.push_back(BindByName);

        ResourceName * resName = m_namesPool->getElement();
        new (resName) ResourceName();

        sprintf_s(resName->name, IR_RESOURCENAME_MAX*sizeof(char), "%s", name);

        m_constBufPSNames.push_back(resName);

        return constantBuf;
    }

    void Pass::setSizeScale(float scaleX, float scaleY)
    {
        m_scaleWidth = scaleX;
        m_scaleHeight = scaleY;
    }

    void Pass::setSize(int sizeX, int sizeY)
    {
        m_baseWidth = sizeX;
        m_baseHeight = sizeY;
    }

    void Pass::deriveSize(int effectInputWidth, int effectInputHeight)
    {
        m_width = (m_baseWidth == Pass::SetAsEffectInputSize) ? effectInputWidth : m_baseWidth;
        m_height = (m_baseHeight == Pass::SetAsEffectInputSize) ? effectInputHeight: m_baseHeight;

        m_width = (int)(m_scaleWidth * m_width);
        m_height = (int)(m_scaleHeight * m_height);
    }

    Pass::Pass(Pool<ResourceName>* namesPool, VertexShader * vertexShader, PixelShader * pixelShader, int width, int height, FragmentFormat * mrtFormats, int numMRTChannels) :
        m_namesPool(namesPool),
        m_vertexShader(vertexShader),
        m_pixelShader(pixelShader),
        m_baseWidth(width),
        m_baseHeight(height),
        m_dataOutMRTTotal(numMRTChannels)
    {
        for (int i = 0; i < numMRTChannels; ++i)
            m_mrtChannelFormats.push_back(mrtFormats[i]);
    }

    Pass::~Pass()
    {
        for (size_t i = 0, iend = m_nameSrc.size(); i < iend; ++i)
        {
            if (m_nameSrc[i])
            {
                //m_nameSrc[i]->~ResourceName();
                m_namesPool->deleteElement(m_nameSrc[i]);
            }
        }
        for (size_t i = 0, iend = m_samplerNames.size(); i < iend; ++i)
        {
            if (m_samplerNames[i])
            {
                //m_samplerNames[i]->~ResourceName();
                m_namesPool->deleteElement(m_samplerNames[i]);
            }
        }
        for (size_t i = 0, iend = m_constBufPSNames.size(); i < iend; ++i)
        {
            if (m_constBufPSNames[i])
            {
                //m_constBufNames[i]->~ResourceName();
                m_namesPool->deleteElement(m_constBufPSNames[i]);
            }
        }
        for (size_t i = 0, iend = m_constBufVSNames.size(); i < iend; ++i)
        {
            if (m_constBufVSNames[i])
            {
                //m_constBufNames[i]->~ResourceName();
                m_namesPool->deleteElement(m_constBufVSNames[i]);
            }
        }
    }

    size_t Pass::getDataSrcNum() const
    {
        return m_dataSources.size();
    }
    size_t Pass::getDataOutNum() const
    {
        return m_dataOut.size();
    }

    const DataSource * Pass::getDataSrc(size_t idx) const
    {
        return m_dataSources[idx];
    }
    const Texture * Pass::getDataOut(size_t idx) const
    {
        return m_dataOut[idx];
    }

    DataSource * Pass::getDataSrc(size_t idx)
    {
        return m_dataSources[idx];
    }
    Texture * Pass::getDataOut(size_t idx)
    {
        return m_dataOut[idx];
    }

    const char * Pass::getNameSrc(size_t idx) const
    {
        return m_nameSrc[idx]->name;
    }
    int Pass::getSlotSrc(size_t idx) const
    {
        return m_slotSrc[idx];
    }
    int Pass::getMRTChannelSrc(size_t idx) const
    {
        return m_mrtChannelSrc[idx];
    }
    int Pass::getMRTChannelOut(size_t idx) const
    {
        return m_mrtChannelOut[idx];
    }

    FragmentFormat Pass::getMRTChannelFormat(size_t mrtIdx) const
    {
        return m_mrtChannelFormats[mrtIdx];
    }
    int Pass::getMRTChannelsTotal() const
    {
        return m_dataOutMRTTotal;
    }

    size_t Pass::getSamplersNum() const
    {
        return m_samplers.size();
    }
    Sampler * Pass::getSampler(size_t idx)
    {
        return m_samplers[idx];
    }
    int Pass::getSamplerSlot(size_t idx) const
    {
        return m_samplerSlots[idx];
    }
    const char * Pass::getSamplerName(size_t idx) const
    {
        // TODO: add debug assertion / nullptr check
        return m_samplerNames[idx]->name;
    }

    size_t Pass::getConstBufsVSNum() const
    {
        return m_constBufsVS.size();
    }
    ConstantBuf * Pass::getConstBufVS(size_t idx)
    {
        return m_constBufsVS[idx];
    }
    int Pass::getConstBufVSSlot(size_t idx) const
    {
        return m_constBufVSSlots[idx];
    }
    const char * Pass::getConstBufVSName(size_t idx) const
    {
        // TODO: add debug assertion / nullptr check
        return m_constBufVSNames[idx]->name;
    }

    size_t Pass::getConstBufsPSNum() const
    {
        return m_constBufsPS.size();
    }
    ConstantBuf * Pass::getConstBufPS(size_t idx)
    {
        return m_constBufsPS[idx];
    }
    int Pass::getConstBufPSSlot(size_t idx) const
    {
        return m_constBufPSSlots[idx];
    }
    const char * Pass::getConstBufPSName(size_t idx) const
    {
        // TODO: add debug assertion / nullptr check
        return m_constBufPSNames[idx]->name;
    }

    void Pass::replaceDataSrc(size_t idx, DataSource * newDataSrc, int mrtChannel)
    {
        m_dataSources[idx] = newDataSrc;
        m_mrtChannelSrc[idx] = mrtChannel;
    }
}
}
