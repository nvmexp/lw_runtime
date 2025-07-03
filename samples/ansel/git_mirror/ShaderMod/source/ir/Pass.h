#pragma once

#include "DataSource.h"
#include "SpecializedPool.h"
#include "PipelineStates.h"

#include <vector>

class ConstantBuf;
class PixelShader; 
class Sampler;
class Texture;
class VertexShader;

namespace shadermod
{
namespace ir
{

    // TODO: add proper const get/setters
    class Pass : public DataSource
    {
    public:

        static const int SetAsEffectInputSize = -1;
        static const int BindByName = -1;

        virtual DataType getDataType() const override { return DataType::kPass; }

        DataSource * addDataSource(DataSource * dataSrc, int slot, int mrtChannel = 0);
        DataSource * addDataSource(DataSource * dataSrc, const char * name, int mrtChannel = 0);
        Texture * addDataOut(Texture * dataOut, int mrtChannel = 0);
        Sampler * addSampler(Sampler * sampler, int slot);
        Sampler * addSampler(Sampler * sampler, const char * name);
        ConstantBuf * addConstantBufferVS(ConstantBuf * constantBuf, int slot);
        ConstantBuf * addConstantBufferVS(ConstantBuf * constantBuf, const char * name);
        ConstantBuf * addConstantBufferPS(ConstantBuf * constantBuf, int slot);
        ConstantBuf * addConstantBufferPS(ConstantBuf * constantBuf, const char * name);

        void setSizeScale(float scaleX, float scaleY);
        void setSize(int sizeX, int sizeY);
        void deriveSize(int effectInputWidth, int effectInputHeight);

        Pass::Pass(Pool<ResourceName>* namesPool, VertexShader * vertexShader, PixelShader * pixelShader, int baseWidth, int baseHeight, FragmentFormat * mrtFormats, int numMRTChannels = 1);
        Pass::~Pass();

        size_t                  getDataSrcNum() const;
        size_t                  getDataOutNum() const;

        const DataSource *      getDataSrc(size_t idx) const;
        const Texture *         getDataOut(size_t idx) const;

        DataSource *            getDataSrc(size_t idx);
        Texture *               getDataOut(size_t idx);

        int                     getSlotSrc(size_t idx) const;
        const char *            getNameSrc(size_t idx) const;
        int                     getMRTChannelSrc(size_t idx) const;
        int                     getMRTChannelOut(size_t idx) const;

        FragmentFormat          getMRTChannelFormat(size_t mrtIdx) const;
        int                     getMRTChannelsTotal() const;

        size_t                  getSamplersNum() const;
        Sampler *               getSampler(size_t idx);
        int                     getSamplerSlot(size_t idx) const;
        const char *            getSamplerName(size_t idx) const;

        size_t                  getConstBufsVSNum() const;
        ConstantBuf *           getConstBufVS(size_t idx);
        int                     getConstBufVSSlot(size_t idx) const;
        const char *            getConstBufVSName(size_t idx) const;

        size_t                  getConstBufsPSNum() const;
        ConstantBuf *           getConstBufPS(size_t idx);
        int                     getConstBufPSSlot(size_t idx) const;
        const char *            getConstBufPSName(size_t idx) const;

        void replaceDataSrc(size_t idx, DataSource * newDataSrc, int mrtChannel = 0);

        int                             m_width, m_baseWidth;
        int                             m_height, m_baseHeight;
        PixelShader *                   m_pixelShader;
        VertexShader *                  m_vertexShader;

        float                           m_scaleWidth = 1.0f;
        float                           m_scaleHeight = 1.0f;
    
        RasterizerState                 m_rasterizerState;
        DepthStencilState               m_depthStencilState;
        AlphaBlendState                 m_alphaBlendState;

        std::vector<int>                m_samplerSlots;
        std::vector<ResourceName *>     m_samplerNames;
        std::vector<Sampler *>          m_samplers;

        std::vector<ConstantBuf *>      m_constBufsVS;
        std::vector<int>                m_constBufVSSlots;
        std::vector<ResourceName *>     m_constBufVSNames;

        std::vector<ConstantBuf *>      m_constBufsPS;
        std::vector<int>                m_constBufPSSlots;
        std::vector<ResourceName *>     m_constBufPSNames;

        std::vector<int>                m_slotSrc;
        std::vector<ResourceName *>     m_nameSrc;
        std::vector<int>                m_mrtChannelSrc;
        std::vector<int>                m_mrtChannelOut;
        std::vector<DataSource *>       m_dataSources;
        std::vector<Texture *>          m_dataOut;
        std::vector<FragmentFormat>     m_mrtChannelFormats;
        int                             m_dataOutMRTTotal;

    protected:

        Pool<ResourceName> *            m_namesPool;
    };

}
}
