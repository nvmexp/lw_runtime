#pragma once

class AnselBufferColwerter;
class AnselBufferDepth;
class AnselBufferHudless;
struct EffectsInfo;

class BufferTestingOptionsFilter
{
public:
    BufferTestingOptionsFilter(const EffectsInfo& effInfo,
        AnselBufferColwerter& renderBufferColwerter,
        AnselBufferDepth& depthBuf,
        AnselBufferHudless& hudlessBuf);

    void toggleAllow(bool allow);
    bool checkFilter(bool depthBufferUsed, bool hudlessBufferUsed);

private:
    const EffectsInfo& m_effectsInfo;

    AnselBufferColwerter& m_renderBufferColwerter;
    AnselBufferDepth& m_depthBuf;
    AnselBufferHudless& m_hudlessBuf;
        
    bool m_allow = false;
};
