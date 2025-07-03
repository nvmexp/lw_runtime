#pragma once

#include <Windows.h>
#include <d3d11.h>

#include "Ansel.h"
#include "Log.h"
#include "CommonStructs.h"

#include <map>
#include <array>
#include <set>
#include <queue>
#include <string>

class AnselBufferInterface
{
public:

    ID3D11Device *          m_serverD3D11Device = nullptr;
    ID3D11DeviceContext *   m_serverImmediateContext = nullptr;

    HANSELCLIENT m_hClient = nullptr;
    ClientFunctionTable * m_pClientFunctionTable = nullptr;

    // TODO: probably add flags (isAcquired etc.) into the AnselRelease
    //  since it would be common if two different buffers will share same hClientResource
    //  this will remove the need to pass renderbuffers to acq/rel
    std::map<HCLIENTRESOURCE, AnselResource *> * m_handleToAnselResource = nullptr;

    AnselBufferInterface()
    {
    }

    void setMapPointer(std::map<HCLIENTRESOURCE, AnselResource *> * handleToAnselResource)
    {
        m_handleToAnselResource = handleToAnselResource;
    }

    void setClientData(HANSELCLIENT & hClient, ClientFunctionTable * pClientFunctionTable, std::map<HCLIENTRESOURCE, AnselResource *> * handleToAnselResource)
    {
        setMapPointer(handleToAnselResource);
        m_pClientFunctionTable = pClientFunctionTable;
        m_hClient = hClient;
    }

    void setServerGAPIData(ID3D11Device * d3d11Device, ID3D11DeviceContext * immediateContext)
    {
        m_serverD3D11Device = d3d11Device;
        m_serverImmediateContext = immediateContext;
    }

    virtual HRESULT copyClientResource(
        HCLIENTRESOURCE hResource,
        ANSEL_TRANSFER_OP op,
        DWORD subResourceIndex,
        DWORD acquireKey,
        DWORD releaseKey
        ) = 0;

    virtual HRESULT sendClientResource(
        HCLIENTRESOURCE hResource,
        ANSEL_TRANSFER_OP op,
        DWORD subResourceIndex,
        DWORD acquireKey,
        DWORD releaseKey
        ) = 0;

    virtual HRESULT getClientResourceInfo(
        HCLIENTRESOURCE hResource,
        AnselClientResourceInfo * pResourceInfo
        )
    {
        return m_pClientFunctionTable->GetClientResourceInfo(m_hClient, hResource, pResourceInfo);
    }


    virtual HRESULT requestClientResource(
        HCLIENTRESOURCE hResource,
        void * pToServerPrivateData,
        void * pToClientPrivateData
        )
    {
        return m_pClientFunctionTable->RequestClientResource(m_hClient, hResource, pToServerPrivateData, pToClientPrivateData);
    }

    // TODO avoroshilov: should it actually be there? Probably move to Server
    virtual AnselResource * lookupAnselResource(
        HCLIENTRESOURCE hClientResource
        )
    {
        AnselResource * rtn = nullptr;

        if (!m_handleToAnselResource)
            return nullptr;

        std::map<HCLIENTRESOURCE, AnselResource *>::iterator it = m_handleToAnselResource->find(hClientResource);
        if (it != m_handleToAnselResource->end())
        {
            AnselResource * pResViewData = it->second;
            if (pResViewData)
            {
                rtn = pResViewData;
            }
        }
        return rtn;
    }

    virtual void storeAnselResource(
        HCLIENTRESOURCE hClientResource,
        AnselResource * pAnselResource
        )
    {
        (*m_handleToAnselResource)[hClientResource] = pAnselResource;
    }
};

class AnselD3D11BufferInterface : public AnselBufferInterface
{
public:
    AnselD3D11BufferInterface()
    {
    }

    HRESULT copyClientResource(
        HCLIENTRESOURCE hResource,
        ANSEL_TRANSFER_OP op,
        DWORD subResourceIndex,
        DWORD acquireKey,
        DWORD releaseKey
        ) final
    {
        return m_pClientFunctionTable->CopyClientResource(m_hClient, hResource, op, subResourceIndex, acquireKey, releaseKey);
    }

    HRESULT sendClientResource(
        HCLIENTRESOURCE hResource,
        ANSEL_TRANSFER_OP op,
        DWORD subResourceIndex,
        DWORD acquireKey,
        DWORD releaseKey
        ) final
    {
        return m_pClientFunctionTable->SendClientResource(m_hClient, hResource, op, subResourceIndex, acquireKey, releaseKey);
    }

    void clearLwrrentBuffers();
        
    // Current graphics resources (for buffer extraction)
    HCLIENTRESOURCE m_lwrrentDepthBuffer;
    HCLIENTRESOURCE m_lwrrentHDR;
    HCLIENTRESOURCE m_lwrrentHUDless;
};

class AnselD3D12BufferInterface : public AnselBufferInterface
{
public:

    ANSEL_EXEC_DATA * m_pLwrrentExecData = nullptr;

    AnselD3D12BufferInterface()
    {
    }

    void setLwrrentExecDataPointer(ANSEL_EXEC_DATA * pLwrrentExecData)  { m_pLwrrentExecData = pLwrrentExecData; }

    virtual HRESULT copyClientResource(
        HCLIENTRESOURCE hResource,
        ANSEL_TRANSFER_OP op,
        DWORD subResourceIndex,
        DWORD acquireKey,
        DWORD releaseKey
        ) override
    {
        return m_pClientFunctionTable->CopyClientResource12(m_pLwrrentExecData->hExelwtionContext, m_hClient, hResource, op, subResourceIndex, acquireKey, releaseKey);
    }

    virtual HRESULT sendClientResource(
        HCLIENTRESOURCE hResource,
        ANSEL_TRANSFER_OP op,
        DWORD subResourceIndex,
        DWORD acquireKey,
        DWORD releaseKey
        ) override
    {
        return m_pClientFunctionTable->SendClientResource12(m_pLwrrentExecData->hExelwtionContext, m_hClient, hResource, op, subResourceIndex, acquireKey, releaseKey);
    }
};

class BufferStats
{
public:
    using Id = std::pair<HCLIENTRESOURCE, UINT32>;
    static constexpr Id s_NullId = { nullptr, 0 };

    BufferStats(const Id& id, size_t bindNum);

    static const Id genId(HCLIENTRESOURCE resource, UINT32 copyIdx) {
        return { resource, copyIdx };
    }

    static const HCLIENTRESOURCE getResFromId(const Id& id) {
        return id.first;
    }

    static UINT32 getInstFromId(const Id& id) {
        return id.second;
    }

    bool operator==(const BufferStats& rhs) {
        return this->m_id == rhs.m_id;
    }

    bool operator<(const BufferStats& rhs) const {

        // First compare score
        if (this->m_score < rhs.m_score)
        {
            return true;
        }
        if (this->m_score > rhs.m_score)
        {
            return false;
        }

        // In case of score tie, prefer the one that has a larger viewport
        if (this->getViewport() < rhs.getViewport())
        {
            return true;
        }
        if (this->getViewport() > rhs.getViewport())
        {
            return false;
        }

        // In case of viewport tie, prefer the one that with the largest number of draws
        if (this->getDraws() < rhs.getDraws())
        {
            return true;
        }
        if (this->getDraws() > rhs.getDraws())
        {
            return false;
        }

        // Otherwise, order based on Id
        if (this->m_id.first < rhs.m_id.first)
        {
            return true;
        }

        if (this->m_id.first > rhs.m_id.first)
        {
            return false;
        }

        return this->m_id.second < rhs.m_id.second;
    }

    void update(const AnselDeviceStates& devState, size_t drawNum);
    void clear();
    void processMappedStats();
    
    size_t callwlateVPScore(size_t weightEqual, size_t weightRatio) const;

    void setScore(size_t score)     { m_score = score; }
    size_t getScore() const         { return m_score; }
    size_t getBindNum() const       { return m_bindNum; }
    size_t getDraws() const         { return m_draws; }
    size_t getDepths() const        { return m_depths; }
    size_t getBlends() const        { return m_blends; }
    size_t getStencils() const      { return m_stencils; }
    size_t getViewportsSize() const { return m_viewports.size(); }

    size_t getRtwMask() const   {
        return m_cachedRtwMask;
    }
    std::pair<FLOAT, FLOAT> getBackbuf() const {
        assert(m_cachedBackbuf.first != 0.0f && m_cachedBackbuf.second != 0.0f);
        return m_cachedBackbuf;
    }
    std::pair<FLOAT, FLOAT> getViewport() const {
        assert(m_cachedViewport.first != 0.0f && m_cachedViewport.second != 0.0f);
        return m_cachedViewport;
    }
    std::pair<FLOAT, FLOAT> getViewportTop() const {
        return m_cachedViewportTop;
    }

    const Id getId() const { return m_id; }
    const HCLIENTRESOURCE getIdResource() const { return getResFromId(m_id); }
    const UINT32 getIdInstance() const { return getInstFromId(m_id); }

private:
    size_t m_score = 0;

    size_t m_bindNum = 0;
    size_t m_draws = 0;
    size_t m_depths = 0;
    size_t m_blends = 0;
    size_t m_stencils = 0;

    size_t m_cachedRtwMask = 0;
    std::pair<FLOAT, FLOAT> m_cachedBackbuf = { 0.0f, 0.0f };
    std::pair<FLOAT, FLOAT> m_cachedViewport = { 0.0f, 0.0f };
    std::pair<FLOAT, FLOAT> m_cachedViewportTop = { 0.0f, 0.0f };

    std::map<UINT8, size_t> m_rtwMasks;
    std::map<std::pair<DWORD, DWORD>, size_t> m_backbufs;
    std::map<std::pair<FLOAT, FLOAT>, size_t> m_viewports;
    std::map<std::pair<FLOAT, FLOAT>, size_t> m_viewportTops;

    Id m_id;
};

class AnselBuffer
{
public:
    enum class Type {
        kPresent,
        kFinal,     // If specific hints are used, alternative to presentable
        kDepth,
        kHDR,
        kHudless,

        kNUM_ENTRIES
    };

    virtual ~AnselBuffer() {}

    bool m_didCopyFromClient = false;
    bool m_needCopyToClient = false;
    
    AnselBufferInterface * m_pBufferInterface = nullptr;

    // Initial key is always 0
    DWORD incWaitKey(DWORD waitKey)
    {
        return waitKey + 1;
    }

    // This flag shows that resource could be grabbed at intermediate stages and could be shared
    //  so that it needs individual resource copy in order to prevent some other buffer overwriting
    //  its contents.
    //  e.g. HUDless buffer - it could be the same presentable surface, and since it could later be
    //  overwritten by the presentable buffer contents, it requires separate Texture2D/SRV/RTVs
    bool m_requireResourceDuplicate = false;
    // If resource duplicate is actually filled this frame
    bool m_resourceDuplicateValid = false;
    bool m_bIlwalidResSize = false;

    const AnselResourceData * getValidResourceData()
    {
        if (m_resourceDuplicateValid)
        {
            return &m_resourceCopy;
        }
        else
        {
            if (m_pAnselResource)
                return static_cast<AnselResourceData *>(&m_pAnselResource->toServerRes);
            else
                return nullptr;
        }
    }

    AnselResourceData m_resourceCopy;
    HRESULT createResourceCopy(
        DWORD width,
        DWORD height,
        DWORD sampleCount,
        DWORD sampleQuality,
        DXGI_FORMAT format,
        bool needsSRV = false,
        bool needsRTV = false
        );

    static bool isHUDlessColor(const AnselDeviceStates &deviceStates);
    bool isDepthBuf() const { return m_type == Type::kDepth; }
    bool isBufferReadyToUse() const { return m_isReadyToUse; }

    bool isForced() const { return m_isForced; }
    void setForced(HCLIENTRESOURCE clientResource);

    void setInternalName(const char * internalName) { m_internalName = internalName; }
    const std::string getInternalName() const { return m_internalName; }

    void setClientResource(HCLIENTRESOURCE clientResource)
    {
        m_clientResource = clientResource;
    }
    void setAnselResource(AnselResource * anselResource)
    {
        m_pAnselResource = anselResource;
    }
    HCLIENTRESOURCE getClientResource() const { return m_clientResource; }
    AnselResource * getAnselResource() const { return m_pAnselResource; }

    bool isToClientGraphicsResourceNeeded()
    {
        return m_isWritable;
    }
    bool isToServerGraphicsResourceNeeded()
    {
        // ToServer resource is needed in both RO and RW cases
        // it is not needed in WO mode, but we don't have such
        return true;
    }

    bool isToClientCopyNeeded()
    {
        return m_isWritable;
    }

    void resetStatus()
    {
        if (!m_pAnselResource)
        {
            return;
        }

        m_didCopyFromClient = false;
        resetSurfaceData();
    }
    void resetSurfaceData()
    {
        m_pAnselResource = nullptr;
        m_clientResource = nullptr;
    }

    void checkIfBufferAlreadyAcquired(bool * isBufferToServerAlreadyAcquired, bool * isBufferToClientAlreadyAcquired);
    void setReleaseBufferAlreadyAcquired();
    
    HRESULT AnselBuffer::acquireInternal(
        bool isAcqNeeded,
        AnselSharedResourceData * sharedResourceData,
        bool * pAcqFlag,
        DWORD * pAcqKey,
        const char * logPostfix = nullptr
        );
    HRESULT acquire(DWORD subResIndex);

    HRESULT AnselBuffer::releaseInternal(
        bool isRelNeeded,
        AnselSharedResourceData * sharedResourceData,
        bool * pAcqFlag,
        DWORD * pRelKey,
        const char * logPostfix = nullptr
        );
    HRESULT release(bool forceNoCopy = false);

    // This function checks if server-side graphics resources need to be created
    //  there are two kind of such resrources: ToServer and ToClient
    //  ToServer is a resource that will be readable from the server, e.g. used as a source of data
    //  ToClient will be used to substitute the data that client-side will then work with
    //  Typical usecases: depth requires only ToServer, as it is RO; presentable is RW, requires both
    //  HDR and HUDless are also potentially RW since mid-pipe shader injection will require their modification
    HRESULT checkCreateServerGraphicsResources();

    HRESULT copyResource(DWORD subResIndex);

    virtual HRESULT selectBuffer(HCLIENTRESOURCE clientResource);

    void destroy();

    // Functions for Stats mechanism
    void setStatsEn(bool enable);
    bool useStats() const;
    void bufBound(HCLIENTRESOURCE resource);
    void bufCleared(HCLIENTRESOURCE resource);
    void removeBuf(HCLIENTRESOURCE resource);
    bool compareAgainstSelBuf(HCLIENTRESOURCE resource) const;
    bool isResourceTracked(HCLIENTRESOURCE resource) const;
    void resolveStats();
    void resolveStatsDebugSelectBuf();
    void addStats(HCLIENTRESOURCE resource, const AnselDeviceStates& devState);

    HCLIENTRESOURCE getSelectedResource() const {
        return BufferStats::getResFromId(m_selectedBuf);
    }

    void clearSelectedBuf();

    size_t getBufferSelect() const;
    void setBufferSelect(size_t buffer);

    virtual size_t callwlateScore(BufferStats &stats) const;

protected:
    AnselBuffer() {}

    static const std::vector<std::pair<std::wstring, uint32_t>> parseWeightStr(const std::wstring& weightsStr);

    std::string m_internalName;
    Type m_type;

    // Flags whether buffer is writable by Ansel
    bool m_isWritable = false;

    // Data members for Stats mechanism
    bool m_useStats = false;
    std::map<HCLIENTRESOURCE, size_t> m_BufStatsDrawCount;      // Count how many times a resource was drawn against
    std::map<const BufferStats::Id, BufferStats> m_BufStats;    // Container which gathers stats during bind/draw calls
    std::vector<BufferStats> m_BufStatsOrdered;                 // Container which orders buffer stats after resolve
    size_t m_orderedBufSelect = 0;                              // Buffer idx that user explicitly requested using BufferTestingOptions filter
    BufferStats::Id m_selectedBuf;

    // Track all the instances of this buffer, where each bind/clear of a frame signifies a new instance
    std::map<HCLIENTRESOURCE, UINT32> m_resClearTracker;

private:
    static HRESULT refreshResourceIfNeeded(
        ID3D11Device * d3d11Device,
        AnselResourceData * resourceData,
        DWORD width,
        DWORD height,
        DWORD sampleCount,
        DWORD sampleQuality,
        DXGI_FORMAT format,
        bool needsSRV = false,
        bool needsRTV = false
    );

    HRESULT copyResourceInternal(DWORD subResIndex);

    void resetCopyFlags()
    {
        m_didCopyFromClient = false;
        m_needCopyToClient = false;
        m_resourceDuplicateValid = false;
    }

    void clearPerFrameData();

    // States
    AnselResource * m_pAnselResource = nullptr;
    HCLIENTRESOURCE m_clientResource = nullptr;

    // Flag shows whether the buffer was correctly acquired
    bool m_isReadyToUse = false;

    // Flag that signals if hinting API was used on this buffer
    bool m_isForced = false;

    DWORD m_subresourceIndex = 0;
};

class AnselBufferPresent : public AnselBuffer
{
public:
    AnselBufferPresent()
    {
        m_type = AnselBuffer::Type::kPresent;
        m_internalName = "present";
        m_isWritable = true;
    }
};

class AnselBufferFinal : public AnselBuffer
{
public:
    AnselBufferFinal()
    {
        m_type = AnselBuffer::Type::kFinal;
        m_internalName = "finalColor";
        m_requireResourceDuplicate = true;
        m_isWritable = true;
    }
};

class AnselBufferDepth : public AnselBuffer
{
public:
    AnselBufferDepth()
    {
        m_type = AnselBuffer::Type::kDepth;
        m_internalName =  "depth";
    }

    HRESULT selectBuffer(HCLIENTRESOURCE clientResource) override;

    HRESULT checkBuffer(HCLIENTRESOURCE clientResource, const AnselDeviceStates& deviceStates,
        DWORD width, DWORD height, bool* result) const;

    void setViewportChecksEn(bool enable);
    bool useViewportChecks() const;

    void setViewportScalingEn(bool enable);
    bool useViewportScaling() const;

    void setWeights(const std::wstring& weightsStr);

private:
    static bool isOpaqueDraw(const AnselDeviceStates &deviceStates);

    // Weights used for Depth Buffer Analysis scoring
    uint32_t m_weightBlend = 1;
    uint32_t m_weightStencil = 1;
    uint32_t m_weightDepthOverBlend = 1;
    uint32_t m_weightDepthOverStencil = 1;
    uint32_t m_weightRtwColorMask = 1;
    uint32_t m_weightVPEqualsBB = 5;
    uint32_t m_weightVPMatchesBBRatio = 4;
    uint32_t m_weightVPTopZero = 5;

    size_t callwlateScore(BufferStats &stats) const override;

    bool m_useViewportChecks = false;
    bool m_useViewportScaling = false;
};

class AnselBufferHDR : public AnselBuffer
{
public:
    AnselBufferHDR()
    {
        m_type = AnselBuffer::Type::kHDR;
        m_internalName = "hdr";
    }

    HRESULT checkBuffer(HCLIENTRESOURCE clientResource, DWORD width, DWORD height, bool* result);

private:
    static bool isFormatSupported(DWORD format);
    static std::set<DXGI_FORMAT> s_supportedFormats;
};

class AnselBufferHudless : public AnselBuffer
{
public:
    AnselBufferHudless()
    {
        m_type = AnselBuffer::Type::kHudless;
        m_internalName = "hudless";
        m_requireResourceDuplicate = true;
    }

    HRESULT checkBuffer(HCLIENTRESOURCE clientResource, const AnselDeviceStates& deviceStates, bool* result);
    bool copyHudless(HCLIENTRESOURCE clientResource);

    size_t getCompareDrawNum() const;
    void setCompareDrawNum(size_t compareDrawNum);

    void setSingleRTV(bool enable);
    bool useSingleRTV() const;

    void setRestrictFormats(bool enable);
    bool useRestrictFormats() const;

    void setWeights(const std::wstring& weightsStr);

    static bool isFormatSupported(DWORD format);

private:
    static const std::set<DXGI_FORMAT> s_supportedFormats;

    uint32_t m_weightBind = 10;
    uint32_t m_weightDrawDepthDiff = 10;
    uint32_t m_weightZeroStencil = 100;
    uint32_t m_weightVPEqualsBB = 1000;
    uint32_t m_weightVPMatchesBBRatio = 4;
    uint32_t m_weightSingleViewport = 1;

    size_t callwlateScore(BufferStats &stats) const override;

    bool m_useSingleRTV = false;
    bool m_restrictFormats = false;

    size_t m_compareDrawNum = 0;
};

class AnselBufferDB
{
public:
    AnselBufferPresent& Present() const;
    AnselBufferFinal& Final() const;
    AnselBufferDepth& Depth() const;
    AnselBufferHDR& HDR() const;
    AnselBufferHudless& Hudless() const;

    bool setBuffersInterfaceIfNeeded(AnselBufferInterface * pBufferInterface);
    bool clearAnselResource(const AnselResource* pAnselResource);
    bool clearClientResource(HCLIENTRESOURCE clientResource);
    void release();
    void destroy();

    ~AnselBufferDB();

private:
    AnselBuffer* GetBuf(AnselBuffer::Type type) const;

    std::array<AnselBuffer*, static_cast<size_t>(AnselBuffer::Type::kNUM_ENTRIES)> m_bufs = {
        new AnselBufferPresent(),
        new AnselBufferFinal(),
        new AnselBufferDepth(),
        new AnselBufferHDR(),
        new AnselBufferHudless()
    };
};

class AnselRenderBufferReleaseHelper
{
public:

    AnselRenderBufferReleaseHelper(AnselBuffer * bufferToTrack):
        m_bufferToTrack(bufferToTrack)
    {
    }
    ~AnselRenderBufferReleaseHelper()
    {
        if (m_bufferToTrack)
        {
            HRESULT status = S_OK;
            if (!SUCCEEDED(status = m_bufferToTrack->release(true)))
            {
                LOG_FATAL("Emergency release failed in helper [%s]", m_bufferToTrack->getInternalName().c_str());
            }
        }
    }

private:
    AnselBuffer * m_bufferToTrack = nullptr;
};
