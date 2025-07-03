
#include "BufferTestingOptionsFilter.h"
#include "EffectsInfo.h"
#include "RenderBuffer.h"
#include "RenderBufferColwert.h"
#include "ShaderModMultiPass.h"

BufferTestingOptionsFilter::BufferTestingOptionsFilter(const EffectsInfo& effInfo,
    AnselBufferColwerter& renderBufferColwerter,
    AnselBufferDepth& depthBuf,
    AnselBufferHudless& hudlessBuf)
: m_effectsInfo(effInfo)
, m_renderBufferColwerter(renderBufferColwerter)
, m_depthBuf(depthBuf)
, m_hudlessBuf(hudlessBuf)
{
}

void BufferTestingOptionsFilter::toggleAllow(bool allow)
{
    m_allow = allow;
}

bool BufferTestingOptionsFilter::checkFilter(bool depthBufferUsed, bool hudlessBufferUsed)
{
    bool filterOptionsChanges = false;
    if (!m_allow)
    {
        return filterOptionsChanges;
    }

    const shadermod::MultiPassEffect* eff = nullptr;

    // Look through the stack effects to see if BufferTestingOptions.yaml is loaded
    for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectsStack.size(); effIdx < effIdxEnd; ++effIdx)
    {
        if (effIdx >= m_effectsInfo.m_effectSelected.size())
        {
            continue;
        }

        int selectedEffect = m_effectsInfo.m_effectSelected[effIdx];
        if (selectedEffect == 0 || selectedEffect == -1)
        {
            continue;
        }

        // Vertify that selected effect has already been created
        if (m_effectsInfo.m_effectsStack[effIdx] != nullptr &&
            !m_effectsInfo.m_effectRebuildRequired[effIdx] &&
            m_effectsInfo.m_effectsStack[effIdx]->getFxFilename() == L"BufferTestingOptions.yaml")
        {
            eff = m_effectsInfo.m_effectsStack[effIdx];
            break;
        }
    }

    if (eff == nullptr)
    {
        return filterOptionsChanges;
    }

    const shadermod::ir::UserConstantManager& ucm = eff->getUserConstantManager();

    const shadermod::ir::UserConstant* ucVP = ucm.findByName("depthUseViewport");
    const shadermod::ir::UserConstant* ucVPScaling = ucm.findByName("depthUseViewportScaling");
    const shadermod::ir::UserConstant* ucDepthStats = ucm.findByName("depthUseStats");
    const shadermod::ir::UserConstant* ucHudlessStats = ucm.findByName("hudlessUseStats");
    const shadermod::ir::UserConstant* ucHudlessSingleRTV = ucm.findByName("hudlessSingleRTV");
    const shadermod::ir::UserConstant* ucHudlessRestrictFormats = ucm.findByName("hudlessRestrictFormats");
    const shadermod::ir::UserConstant* ucHudlessDraw = ucm.findByName("hudlessDrawCall");
    const shadermod::ir::UserConstant* ucHudlessBuffer = ucm.findByName("hudlessBufferSelect");

    bool useDepthViewport = false;
    bool useDepthViewportScaling = false;
    bool useDepthStats = false;
    if (depthBufferUsed)
    {
        ucVP->getValue(useDepthViewport);
        ucVPScaling->getValue(useDepthViewportScaling);
        ucDepthStats->getValue(useDepthStats);
    }
    else
    {
        ucVP->getDefaultValue(useDepthViewport);
        ucVPScaling->getDefaultValue(useDepthViewportScaling);
        ucDepthStats->getDefaultValue(useDepthStats);
    }

    bool useHudlessStats = false;
    bool useSingleRTV = false;
    bool restrictFormats = false;
    unsigned int hudlessDrawCall = 0;
    unsigned int hudlessBuffer = 0;
    if (hudlessBufferUsed)
    {
        ucHudlessStats->getValue(useHudlessStats);
        ucHudlessSingleRTV->getValue(useSingleRTV);
        ucHudlessRestrictFormats->getValue(restrictFormats);
        ucHudlessDraw->getValue(hudlessDrawCall);
        ucHudlessBuffer->getValue(hudlessBuffer);
    }
    else
    {
        ucHudlessStats->getDefaultValue(useHudlessStats);
        ucHudlessSingleRTV->getDefaultValue(useSingleRTV);
        ucHudlessRestrictFormats->getDefaultValue(restrictFormats);
        ucHudlessDraw->getDefaultValue(hudlessDrawCall);
        ucHudlessBuffer->getDefaultValue(hudlessBuffer);
    }

    // If we're disabling viewport, force disable Viewport Scaling
    if (m_depthBuf.useViewportChecks() != useDepthViewport && !useDepthViewport)
    {
        useDepthViewportScaling = false;
    }
    // If we're enabling Viewport Scaling, force enable viewport
    if (m_depthBuf.useViewportScaling() != useDepthViewportScaling && useDepthViewportScaling)
    {
        useDepthViewport = true;
    }

    // If any toggle was switched, trigger a stack rebuild to ensure settings are applied, and
    // reset the viewport values to zero so that viewport settings don't persist
    if (m_depthBuf.useViewportChecks() != useDepthViewport ||
        m_depthBuf.useViewportScaling() != useDepthViewportScaling ||
        m_depthBuf.useStats() != useDepthStats ||
        m_hudlessBuf.useStats() != useHudlessStats ||
        m_hudlessBuf.useSingleRTV() != useSingleRTV ||
        m_hudlessBuf.useRestrictFormats() != restrictFormats ||
        m_hudlessBuf.getCompareDrawNum() != hudlessDrawCall ||
        m_hudlessBuf.getBufferSelect() != hudlessBuffer)
    {
        filterOptionsChanges = true;
        m_renderBufferColwerter.setPSConstBufDataViewport(0.0f, 0.0f);

        m_hudlessBuf.clearSelectedBuf();
    }

    // Toggle settings based on user's selections in the filter
    m_depthBuf.setViewportChecksEn(useDepthViewport);
    m_depthBuf.setViewportScalingEn(useDepthViewportScaling);
    m_depthBuf.setStatsEn(useDepthStats);

    m_hudlessBuf.setStatsEn(useHudlessStats);
    m_hudlessBuf.setSingleRTV(useSingleRTV);
    m_hudlessBuf.setRestrictFormats(restrictFormats);
    m_hudlessBuf.setCompareDrawNum(hudlessDrawCall);
    m_hudlessBuf.setBufferSelect(hudlessBuffer);

    return filterOptionsChanges;
}
