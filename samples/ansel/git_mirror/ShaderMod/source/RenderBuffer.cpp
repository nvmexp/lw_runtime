#include "Utils.h"
#include "RenderBuffer.h"
#include "ir/TypeColwersions.h"
#include "anselutils/Utils.h"
#include "CommonTools.h"

#include <assert.h>

#define SAFE_RELEASE(x) if (x) (x)->Release(), (x) = nullptr;
#ifdef _DEBUG
#define AnselBufferHandleFailure() __debugbreak(); return status;
#else
#define AnselBufferHandleFailure() return status;
#endif

//#define LOG_BUFFER_EXTRA_ENABLED
#ifdef LOG_BUFFER_EXTRA_ENABLED
#define LOG_BUFFER_EXTRA(logLevel, ...) LOG_##logLevel(LogChannel::kRenderBuffer, __VA_ARGS__);
#else
#define LOG_BUFFER_EXTRA(logLevel, ...)
#endif

#define HEUR_RESIZED_BUFFERS            1
#define HEUR_RESIZED_BUFFERS_ASP_EPS    5e-3f
#define HEUR_RESIZED_BUFFERS_SCALEMIN   0.85f
#define HEUR_RESIZED_BUFFERS_SCALEMAX   10.0f

void AnselD3D11BufferInterface::clearLwrrentBuffers()
{
    m_lwrrentDepthBuffer = nullptr;
    m_lwrrentHDR = nullptr;
    m_lwrrentHUDless = nullptr;
}

const std::vector<std::pair<std::wstring, uint32_t>> AnselBuffer::parseWeightStr(const std::wstring& weightsStr)
{
    std::vector<std::pair<std::wstring, uint32_t>> weights;

    // Loop through the weight-value strings; delimited by a comma
    for(const std::wstring& weightKeyPairStr : lwanselutils::StrSplit(weightsStr, ','))
    {
        // Separate the key/value into two separate strings; delimited by an equals sign
        const std::vector<std::wstring> vecWeightVals = lwanselutils::StrSplit(weightKeyPairStr, '=');

        if (vecWeightVals.size() != 2)
        {
            LOG_ERROR(LogChannel::kRenderBuffer, "RenderBuffer received poorly constructed DRS Weight Settings: %ls", weightKeyPairStr.c_str());
            assert(!"RenderBuffer received poorly constructed DRS Weight Settings");
            continue;
        }

        weights.push_back(std::make_pair(vecWeightVals[0], std::stoul(vecWeightVals[1])));
    }

    return weights;
}


HRESULT AnselBuffer::refreshResourceIfNeeded(
    ID3D11Device * d3d11Device,
    AnselResourceData * resourceData,
    DWORD width,
    DWORD height,
    DWORD sampleCount,
    DWORD sampleQuality,
    DXGI_FORMAT format,
    bool needsSRV,
    bool needsRTV
    )
{
    HRESULT status = S_OK;

    // TODO: making resource type itself use _TYPELESS format will allow to not recreate Texture2D
    //  if type changed only subtly (like RGBA8->sRGBA8 - only SRV recreation needed)
    bool resourcesCreationNeeded =
        (!resourceData->pTexture2D) ||
        (resourceData->width != width) ||
        (resourceData->height != height) ||
        (resourceData->format != format);

    if (!resourcesCreationNeeded)
    {
        return S_OK;
    }

    SAFE_RELEASE(resourceData->pRTV);
    SAFE_RELEASE(resourceData->pSRV);
    SAFE_RELEASE(resourceData->pTexture2D);

    ZeroMemory(resourceData, sizeof(AnselResourceData));

    D3D11_TEXTURE2D_DESC textureDesc;
    ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = format;
    textureDesc.SampleDesc.Count = sampleCount;
    textureDesc.SampleDesc.Quality = sampleQuality;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    if (needsRTV)
    {
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
    }
    else
    {
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    }

    if (!SUCCEEDED(status = shadermod::Tools::CreateTexture2D(d3d11Device, &textureDesc, NULL, &resourceData->pTexture2D)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer, "RenderBuffer: Resource creation: texture creation failed");
        AnselBufferHandleFailure();
    }

    resourceData->width = width;
    resourceData->height = height;
    resourceData->sampleCount = sampleCount;
    resourceData->sampleQuality = sampleQuality;
    resourceData->format = format;

    // Create views
    if (needsSRV)
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
        ZeroMemory(&srvDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
        srvDesc.Format = lwanselutils::getSRVFormatDepth(lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(format)));
        srvDesc.ViewDimension = sampleCount > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MostDetailedMip = 0;
        srvDesc.Texture2D.MipLevels = 1;

        if (!SUCCEEDED(status = d3d11Device->CreateShaderResourceView(resourceData->pTexture2D, &srvDesc, &resourceData->pSRV)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer, "RenderBuffer: Failed to create texture SRV");
            AnselBufferHandleFailure();
        }
    }

    if (needsRTV)
    {
        D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
        memset(&rtvDesc, 0, sizeof(rtvDesc));
        rtvDesc.Format = lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(textureDesc.Format));
        rtvDesc.ViewDimension = textureDesc.SampleDesc.Count > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
        rtvDesc.Texture2D.MipSlice = 0;

        if (!SUCCEEDED(status = d3d11Device->CreateRenderTargetView(resourceData->pTexture2D, &rtvDesc, &resourceData->pRTV)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer, "RenderBuffer: Failed to create texture RTV");
            AnselBufferHandleFailure();
        }
    }

    return S_OK;
}

HRESULT AnselBuffer::createResourceCopy(
    DWORD width,
    DWORD height,
    DWORD sampleCount,
    DWORD sampleQuality,
    DXGI_FORMAT format,
    bool needsSRV,
    bool needsRTV
    )
{
    HRESULT status = S_OK;

    if (!m_pAnselResource)
        return S_OK;

    status = refreshResourceIfNeeded(
        m_pBufferInterface->m_serverD3D11Device,
        &m_resourceCopy,
        width,
        height,
        sampleCount,
        sampleQuality,
        format,
        needsSRV,
        needsRTV
        );

    if (!SUCCEEDED(status))
    {
        LOG_ERROR(LogChannel::kRenderBuffer, "RenderBuffer: Failed to refresh resource copy");
        AnselBufferHandleFailure();
    }

    m_pBufferInterface->m_serverImmediateContext->CopySubresourceRegion(m_resourceCopy.pTexture2D, 0, 0, 0, 0, m_pAnselResource->toServerRes.pTexture2D, 0, 0);

    return S_OK;
}

bool AnselBuffer::isHUDlessColor(const AnselDeviceStates &deviceStates)
{
    // DX12-related preprocessor disabled

    // TODO: original extractor also checked (BindFlags & D3D10_DDI_BIND_RENDER_TARGET) && (BindFlags & D3D10_DDI_BIND_PRESENT)
    //      although with an error

    // TODO: this check fails Witcher 3 HUDless detection
    //      original extractor didn't seem to require this (it checks only m_hRtViewLwrr and
    //      m_ViewInfo.GetItemPtr(MakeHashFromHandle(m_hRtViewLwrr))->hDrvResource
    //      )
    if (!deviceStates.RTZeroIsPresent)
    {
        return FALSE;
    }

    if ((deviceStates.ViewportTopLeftX != 0)
        || (deviceStates.ViewportTopLeftY != 0)
        || (deviceStates.ViewportWidth != deviceStates.backbufferWidth)
        || (deviceStates.ViewportHeight != deviceStates.backbufferHeight))
    {
        return FALSE;
    }

    // TODO: original extractor didn't require alpha write mask
    if ((deviceStates.RenderTargetWriteMask & ANSEL_COLOR_WRITE_ENABLE_ALL) != ANSEL_COLOR_WRITE_ENABLE_ALL)
    {
        return FALSE;
    }

    if (deviceStates.RTZeroFormat >= DXGI_FORMAT_R8G8B8A8_UINT)
    {
        return FALSE;
    }

    if (deviceStates.DepthEnable == TRUE)
    {
        return FALSE;
    }
    if (deviceStates.StencilEnable == TRUE)
    {
        return FALSE;
    }

    const bool isBlendEnabled = (deviceStates.BlendEnable == TRUE);
    const bool isSrcBlendAlpha = (deviceStates.SrcBlend == ANSEL_BLEND_SRC_ALPHA);
    const bool isBlendOpAdd = (deviceStates.BlendOp == ANSEL_BLEND_OP_ADD);
    const bool isSrcBlendAlphaOneOrSrcAlpha = ((deviceStates.SrcBlendAlpha == ANSEL_BLEND_ONE) || (deviceStates.SrcBlendAlpha == ANSEL_BLEND_SRC_ALPHA));

    if (!(isBlendEnabled && isSrcBlendAlpha && isBlendOpAdd && isSrcBlendAlphaOneOrSrcAlpha))
    {
        return FALSE;
    }

    LOG_DEBUG(LogChannel::kRenderBuffer, "Color HUDless");

    return TRUE;
}


void AnselBuffer::checkIfBufferAlreadyAcquired(bool * isBufferToServerAlreadyAcquired, bool * isBufferToClientAlreadyAcquired)
{
    *isBufferToServerAlreadyAcquired = false;
    *isBufferToClientAlreadyAcquired = false;

    if (!m_pAnselResource)
        return;

    if (isToServerGraphicsResourceNeeded() && m_pAnselResource->toServerAcquired)
    {
        *isBufferToServerAlreadyAcquired = true;
    }
    if (isToClientGraphicsResourceNeeded() && m_pAnselResource->toClientAcquired)
    {
        *isBufferToClientAlreadyAcquired = true;
    }
}
void AnselBuffer::setReleaseBufferAlreadyAcquired()
{
    if (!m_pAnselResource)
        return;

    if (isToServerGraphicsResourceNeeded())
    {
        m_pAnselResource->toServerAcquired = false;
    }
    if (isToClientGraphicsResourceNeeded())
    {
        m_pAnselResource->toClientAcquired = false;
    }
}

void AnselBuffer::setForced(HCLIENTRESOURCE clientResource)
{
    // Present buffer isn't forceable
    assert(m_type != Type::kPresent);

    m_isForced = true;
    setClientResource(clientResource);

    LOG_DEBUG(LogChannel::kRenderBuffer, "Forcing %s buffer", m_internalName.c_str());
}

#define DBG_WRITE LOG_DEBUG

HRESULT AnselBuffer::acquireInternal(
    bool isAcqNeeded,
    AnselSharedResourceData * sharedResourceData,
    bool * pAcqFlag,
    DWORD * pAcqKey,
    const char * logPostfix
    )
{
    HRESULT status = S_OK;
    if (isAcqNeeded && !(*pAcqFlag))
    {
        const DWORD acqKey = *pAcqKey;
        if (!SUCCEEDED(status = sharedResourceData->pTexture2DMutex->AcquireSync(acqKey, INFINITE)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer, "Buffer [%s]%s internal acq op (%d) failed", m_internalName.c_str(), logPostfix ? logPostfix : "", acqKey);
            AnselBufferHandleFailure();
        }
        DBG_WRITE(LogChannel::kRenderBuffer, "[%s]%s acquired with %d", m_internalName.c_str(), logPostfix ? logPostfix : "", acqKey);
        (*pAcqFlag) = true;
    }
    return S_OK;
}

HRESULT AnselBuffer::acquire(DWORD subResIndex)
{
    HRESULT status = S_OK;

    if (!m_clientResource)
        return S_OK;

    // We need a check here, since we need to get AnselResource ptr at the very beginning
    if (!SUCCEEDED( checkCreateServerGraphicsResources() ))
    {
        LOG_ERROR(LogChannel::kRenderBuffer, "Buffer [%s] surface check failed", m_internalName.c_str());
        return S_OK;
    }

    DBG_WRITE(LogChannel::kRenderBuffer, "[%s] acquiring %d %d (0x%x, 0x%x)..", m_internalName.c_str(), m_pAnselResource->toServerWaitKey, m_pAnselResource->toClientWaitKey, m_clientResource, m_pAnselResource);

    bool isToServerAlreadyAcquired;
    bool isToClientAlreadyAcquired;
    checkIfBufferAlreadyAcquired(&isToServerAlreadyAcquired, &isToClientAlreadyAcquired);

    // Assumed that if isToServerAlreadyAcquired then the buffer is already copied over
    if (!isToServerAlreadyAcquired)
    {
        m_subresourceIndex = subResIndex;

        // If the surface wasn't copied on notification stage, copy it on acquire
        if (!SUCCEEDED( status = copyResourceInternal(subResIndex) ))
        {
            LOG_ERROR(LogChannel::kRenderBuffer, "Buffer [%s] surface copy failed", m_internalName.c_str());
            AnselBufferHandleFailure();
        }
    }
    else
    {
        m_pAnselResource->toServerAcquired = true;
    }

    if (isToClientAlreadyAcquired)
        m_pAnselResource->toClientAcquired = true;

    if (!m_pAnselResource)
    {
        LOG_ERROR(LogChannel::kRenderBuffer, "Resource wasn't found for buffer [%s]");
        return S_OK;
    }

    if (!SUCCEEDED( status = acquireInternal(
            isToServerGraphicsResourceNeeded(),
            &m_pAnselResource->toServerRes,
            &m_pAnselResource->toServerAcquired,
            &m_pAnselResource->toServerWaitKey,
            "S"
            ) ))
    {
        AnselBufferHandleFailure();
    }

    if (!SUCCEEDED( status = acquireInternal(
            isToClientGraphicsResourceNeeded(),
            &m_pAnselResource->toClientRes,
            &m_pAnselResource->toClientAcquired,
            &m_pAnselResource->toClientWaitKey,
            "C"
            ) ))
    {
        AnselBufferHandleFailure();
    }

    m_isReadyToUse = true;

    return status;
}

HRESULT AnselBuffer::releaseInternal(
    bool isRelNeeded,
    AnselSharedResourceData * sharedResourceData,
    bool * pAcqFlag,
    DWORD * pRelKey,
    const char * logPostfix
    )
{
    HRESULT status = S_OK;
    if (isRelNeeded && (*pAcqFlag))
    {
        *pRelKey = incWaitKey(*pRelKey);
        const DWORD relKey = *pRelKey;

        (*pAcqFlag) = false;
        if (!SUCCEEDED(status = sharedResourceData->pTexture2DMutex->ReleaseSync(relKey)))
        {
            LOG_FATAL(LogChannel::kRenderBuffer, "Buffer [%s]%s internal rel op (%d) failed", m_internalName.c_str(), logPostfix ? logPostfix : "", relKey);
            AnselBufferHandleFailure();
        }
        DBG_WRITE(LogChannel::kRenderBuffer, "[%s]%s released with %d", m_internalName.c_str(), logPostfix ? logPostfix : "", relKey);
    }
    return S_OK;
}

HRESULT AnselBuffer::release(bool forceNoCopy)
{
    HRESULT status = S_OK;

    m_isReadyToUse = false;

    if (m_pAnselResource)
    {
        DBG_WRITE(LogChannel::kRenderBuffer, "[%s] releasing %d %d..", m_internalName.c_str(), m_pAnselResource->toServerWaitKey, m_pAnselResource->toClientWaitKey);
    }

    const bool areResourcesAvailable = m_pAnselResource && m_clientResource;
    if (!SUCCEEDED( status = releaseInternal(
            areResourcesAvailable,
            &m_pAnselResource->toServerRes,
            &m_pAnselResource->toServerAcquired,
            &m_pAnselResource->toServerWaitKey,
            "S"
            ) ))
    {
        AnselBufferHandleFailure();
    }

    // This flag tells if the associated resource was acquired, so that we can avoid redundant copies and also deadlocks
    bool wasToClientAquired = (areResourcesAvailable && m_pAnselResource->toClientAcquired);
    if (!SUCCEEDED( status = releaseInternal(
            areResourcesAvailable,
            &m_pAnselResource->toClientRes,
            &m_pAnselResource->toClientAcquired,
            &m_pAnselResource->toClientWaitKey,
            "C"
            ) ))
    {
        AnselBufferHandleFailure();
    }

    if (m_needCopyToClient && !wasToClientAquired)
    {
        LOG_WARN(LogChannel::kRenderBuffer, "[%s] resource sending requested, but resource already released", m_internalName.c_str());
    }

    // If the surface should be modified for client, copy it after the release
    //  in order for the copy to happen, resources should be available, the buffer should be RW/WO,
    //  copy should be triggered from within the server logic, and also buffer should've been acquired
    if (areResourcesAvailable && !forceNoCopy && isToClientCopyNeeded() && m_needCopyToClient && wasToClientAquired)
    {
        m_needCopyToClient = false;
        const DWORD acqKey = m_pAnselResource->toClientWaitKey;

        m_pAnselResource->toClientWaitKey = incWaitKey(m_pAnselResource->toClientWaitKey);
        const DWORD relKey = m_pAnselResource->toClientWaitKey;
        if (!SUCCEEDED(status = m_pBufferInterface->sendClientResource(m_clientResource, ANSEL_TRANSFER_OP_COPY, m_subresourceIndex, acqKey, relKey)))
        {
            LOG_FATAL(LogChannel::kRenderBuffer, "Resource sending failed for buffer [%s]", m_internalName.c_str());
            AnselBufferHandleFailure();
        }
        DBG_WRITE(LogChannel::kRenderBuffer, "[%s] sent with %d/%d", m_internalName.c_str(), acqKey, relKey);
    }

    if (!areResourcesAvailable && m_needCopyToClient)
    {
        LOG_DEBUG(LogChannel::kRenderBuffer, "Copy requested while resources were unavailable [%s]", m_internalName.c_str());
        m_needCopyToClient = false;
    }

    // If we released the shared resource, we need all the other shared resources
    //  to be marked as released to avoid deadlocks
    setReleaseBufferAlreadyAcquired();

    resetCopyFlags();
    resetSurfaceData();

    return status;
}

HRESULT AnselBuffer::checkCreateServerGraphicsResources()
{
    HRESULT status = S_OK;
    if (!m_clientResource)
    {
        return S_OK;
    }

    AnselResource * pData = m_pBufferInterface->lookupAnselResource(m_clientResource);
    bool bNeedCreateResToServer = false;
    bool bNeedCreateResToClient = false;
    if (!pData)
    {
        bNeedCreateResToServer = isToServerGraphicsResourceNeeded();
        bNeedCreateResToClient = isToClientGraphicsResourceNeeded();
    }
    else
    {
        if (!pData->toServerRes.pTexture2D)
        {
            bNeedCreateResToServer = isToServerGraphicsResourceNeeded();
        }
        if (!pData->toClientRes.pTexture2D)
        {
            bNeedCreateResToClient = isToClientGraphicsResourceNeeded();
        }
    }

    if (bNeedCreateResToServer || bNeedCreateResToClient)
    {
        // Create shared resource to the client resource.
        if (!pData)
        {
            pData = new AnselResource;
            memset(pData, 0, sizeof(AnselResource));
        }

        CREATESHAREDRESOURCEDATA csrDataToServer;
        memset(&csrDataToServer, 0, sizeof(CREATESHAREDRESOURCEDATA));
        csrDataToServer.pSharedResourceData = &pData->toServerRes;
        csrDataToServer.overrideFormat = DXGI_FORMAT_UNKNOWN;

        // Do we need this? Doesn't look like we'd want to render into ToServer surface
        if (isDepthBuf())
        {
            // Depth cannot be rendered to
            csrDataToServer.overrideBindFlags = D3D11_BIND_SHADER_RESOURCE;
        }
        else
        {
            csrDataToServer.overrideBindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
        }

        // TODO: maybe override to this instead?
        //  investigate 'requestClientResource'
        //csrDataToServer.overrideBindFlags = D3D11_BIND_SHADER_RESOURCE;

        // No MSAA for this proxy surface. The contents will be copied to it using a resolve.
        if (!isDepthBuf())
        {
            csrDataToServer.overrideSampleCount = 1;
        }

        CREATESHAREDRESOURCEDATA csrDataToClient;
        memset(&csrDataToClient, 0, sizeof(CREATESHAREDRESOURCEDATA));
        csrDataToClient.pSharedResourceData = &pData->toClientRes;
        csrDataToClient.overrideFormat = DXGI_FORMAT_UNKNOWN;
        csrDataToClient.overrideBindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;

        CREATESHAREDRESOURCEDATA * pCreateResDataToServer = bNeedCreateResToServer ? &csrDataToServer : NULL;
        CREATESHAREDRESOURCEDATA * pCreateResDataToClient = bNeedCreateResToClient ? &csrDataToClient : NULL;
        AnselSharedResourceData * pSharedServerResourceData = (pCreateResDataToServer != NULL) ? pCreateResDataToServer->pSharedResourceData : NULL;
        AnselSharedResourceData * pSharedClientResourceData = (pCreateResDataToClient != NULL) ? pCreateResDataToClient->pSharedResourceData : NULL;

        if (!SUCCEEDED(status = m_pBufferInterface->requestClientResource(m_clientResource, pCreateResDataToServer, pCreateResDataToClient)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer, "Resource requesting failed (0x%x) for buffer [%s]", status, m_internalName.c_str());
            if ((pSharedServerResourceData != NULL && (pSharedServerResourceData->width == 0 || pSharedServerResourceData->height == 0)) ||
                (pSharedClientResourceData != NULL && (pSharedClientResourceData->width == 0 || pSharedClientResourceData->height == 0)))
            {
                m_bIlwalidResSize = true;
                return status;
            }
            else
            {
                AnselBufferHandleFailure();
            }
        }
        m_bIlwalidResSize = false;

        m_pBufferInterface->storeAnselResource(m_clientResource, pData);

        ID3D11Device * d3d11Device = m_pBufferInterface->m_serverD3D11Device;

        // Create views
        if (bNeedCreateResToServer)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
            memset(&srvDesc, 0, sizeof(srvDesc));

            if (isDepthBuf())
            {
                srvDesc.Format = lwanselutils::getSRVFormatDepth(lwanselutils::colwertFromTypelessIfNeeded(DXGI_FORMAT(pData->toServerRes.format)));
            }
            else
            {
                srvDesc.Format = lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(DXGI_FORMAT(pData->toServerRes.format)));
            }

            srvDesc.ViewDimension = pData->toServerRes.sampleCount > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
            if (csrDataToServer.overrideSampleCount == 1)
            {
                srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            }
            srvDesc.Texture2D.MostDetailedMip = 0;
            srvDesc.Texture2D.MipLevels = 1;

            if (!SUCCEEDED(status = d3d11Device->CreateShaderResourceView(pData->toServerRes.pTexture2D, &srvDesc, &pData->toServerRes.pSRV)))
            {
                LOG_ERROR(LogChannel::kRenderBuffer, "Resource SRV creation failed for buffer [%s]", m_internalName.c_str());
                AnselBufferHandleFailure();
            }
        }

        if (bNeedCreateResToClient)
        {
            D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
            memset(&rtvDesc, 0, sizeof(rtvDesc));
            rtvDesc.Format = lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(DXGI_FORMAT(pData->toClientRes.format)));
            rtvDesc.ViewDimension = pData->toClientRes.sampleCount > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
            rtvDesc.Texture2D.MipSlice = 0;

            if (!SUCCEEDED(status = d3d11Device->CreateRenderTargetView(pData->toClientRes.pTexture2D, &rtvDesc, &pData->toClientRes.pRTV)))
            {
                LOG_ERROR(LogChannel::kRenderBuffer, "Resource RTV creation failed for buffer [%s]", m_internalName.c_str());
                AnselBufferHandleFailure();
            }
        }
    }
    m_pAnselResource = pData;
    return status;
}

HRESULT AnselBuffer::copyResourceInternal(DWORD subResIndex)
{
    if (!m_clientResource)
        return S_OK;

    // If we did a copy, we don't need to perform another one
    //  although probably sometimes we would want to do it?
    //  We are no longer limited by the mutex keys
    if (m_didCopyFromClient)
        return S_OK;

    HRESULT status = S_OK;

    const DWORD acqKey = m_pAnselResource->toServerWaitKey;

    m_pAnselResource->toServerWaitKey = incWaitKey(m_pAnselResource->toServerWaitKey);
    const DWORD relKey = m_pAnselResource->toServerWaitKey;

    DBG_WRITE(LogChannel::kRenderBuffer, "[%s] copying %d %d (0x%x, 0x%x)..", m_internalName.c_str(), m_pAnselResource->toServerWaitKey, m_pAnselResource->toClientWaitKey, m_clientResource, m_pAnselResource);
    if (!SUCCEEDED(status = m_pBufferInterface->copyClientResource(m_clientResource, isDepthBuf() ? ANSEL_TRANSFER_OP_COPY : ANSEL_TRANSFER_OP_RESOLVE, subResIndex, acqKey, relKey)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer, "Resource copying failed for buffer [%s]", m_internalName.c_str());
        AnselBufferHandleFailure();
    }
    DBG_WRITE(LogChannel::kRenderBuffer, "[%s] copied with %d/%d", m_internalName.c_str(), acqKey, relKey);
    m_didCopyFromClient = true;

    return S_OK;
}

HRESULT AnselBuffer::copyResource(DWORD subResIndex)
{
    HRESULT status = S_OK;

    if (m_didCopyFromClient || !getClientResource())
    {
        return status;
    }

    LOG_DEBUG(LogChannel::kRenderBuffer, "Copying %s buffer", m_internalName.c_str());

    if (!SUCCEEDED( status = checkCreateServerGraphicsResources() ))
    {
        LOG_FATAL(LogChannel::kRenderBuffer, "Resource checking failed for buffer [%s]", m_internalName.c_str());
        AnselBufferHandleFailure();
    }
    if (!SUCCEEDED( status = copyResourceInternal(subResIndex) ))
    {
        // Logging info inside the function
        AnselBufferHandleFailure();
    }

    if (m_requireResourceDuplicate)
    {
        // TODO avoroshilov: move these into internal acq/rel and unify with main
        bool isToServerAlreadyAcquired;
        bool isToClientAlreadyAcquired;
        checkIfBufferAlreadyAcquired(&isToServerAlreadyAcquired, &isToClientAlreadyAcquired);
        if (isToServerAlreadyAcquired)
        {
            m_pAnselResource->toServerAcquired = true;
        }

        if (!SUCCEEDED( status = acquireInternal(
                isToServerGraphicsResourceNeeded(),
                &m_pAnselResource->toServerRes,
                &m_pAnselResource->toServerAcquired,
                &m_pAnselResource->toServerWaitKey,
                "S"
                ) ))
        {
            AnselBufferHandleFailure();
        }

        createResourceCopy(
            m_pAnselResource->toServerRes.width,
            m_pAnselResource->toServerRes.height,
            m_pAnselResource->toServerRes.sampleCount,
            m_pAnselResource->toServerRes.sampleQuality,
            m_pAnselResource->toServerRes.format
            );
        m_resourceDuplicateValid = true;

        const bool areResourcesAvailable = m_pAnselResource && m_clientResource;
        if (!SUCCEEDED( status = releaseInternal(
                areResourcesAvailable,
                &m_pAnselResource->toServerRes,
                &m_pAnselResource->toServerAcquired,
                &m_pAnselResource->toServerWaitKey,
                "S"
                ) ))
        {
            AnselBufferHandleFailure();
        }
    }

    return S_OK;
}

HRESULT AnselBuffer::selectBuffer(HCLIENTRESOURCE clientResource)
{
    if (m_didCopyFromClient || !clientResource)
    {
        return S_OK;
    }

    setClientResource(clientResource);
    return S_OK;
}

void AnselBuffer::setStatsEn(bool enable)
{
    m_useStats = enable;
}

bool AnselBuffer::useStats() const
{
    return m_useStats;
}

// Track the times a buffer was bound on each frame
void AnselBuffer::bufBound(HCLIENTRESOURCE resource)
{
    if (!useStats() || resource == nullptr)
    {
        return;
    }

    // If a bind happens, make sure an entry in the tracker exists, and create
    // an entry in stats
    const BufferStats::Id statId = BufferStats::genId(resource, m_resClearTracker[resource]);
    m_BufStats.emplace(statId, BufferStats(statId, m_resClearTracker.size()));
}

// Track the times a buffer was cleared on each frame
void AnselBuffer::bufCleared(HCLIENTRESOURCE resource)
{
    if (!useStats() || resource == nullptr)
    {
        return;
    }

    // We don't need to track clears on a buffer that hasn't been bound yet
    if (m_resClearTracker.find(resource) == m_resClearTracker.cend())
    {
        return;
    }

    // If a clear happens, increment the buffer tracker value, and create
    // an entry in stats
    if (m_resClearTracker[resource] < UINT32_MAX)
    {
        const BufferStats::Id newStatId = BufferStats::genId(resource, ++m_resClearTracker[resource]);
        m_BufStats.emplace(newStatId, BufferStats(newStatId, m_resClearTracker.size()));
    }
}

void AnselBuffer::removeBuf(HCLIENTRESOURCE resource)
{
    // Remove all instances of a resource from BufStatsOrdered
    for (auto& itBuf = m_BufStatsOrdered.begin(); itBuf != m_BufStatsOrdered.end(); )
    {
        if (itBuf->getIdResource() == resource)
        {
            itBuf = m_BufStatsOrdered.erase(itBuf);
        }
        else
        {
            ++itBuf;
        }
    }

    // Remove all instances of a resource from BufStats
    for(auto& itBuf = m_BufStats.begin(); itBuf != m_BufStats.end(); )
    {
        if (BufferStats::getResFromId(itBuf->first) == resource)
        {
            itBuf = m_BufStats.erase(itBuf);
        }
        else
        {
            ++itBuf;
        }
    }

    if (BufferStats::getResFromId(m_selectedBuf) == resource)
    {
        m_selectedBuf = BufferStats::s_NullId;
    }

    m_resClearTracker.erase(resource);
}

bool AnselBuffer::compareAgainstSelBuf(HCLIENTRESOURCE resource) const
{
    if (resource == nullptr)
    {
        return false;
    }

    if (m_selectedBuf == BufferStats::s_NullId)
    {
        return false;
    }

    const auto& itTrack = m_resClearTracker.find(resource);
    if (itTrack == m_resClearTracker.end())
    {
        assert(!"Tried to access a buf in compareAgainstSelBuf that wasn't bound!");
        return false;
    }

    const BufferStats::Id statId = BufferStats::genId(resource, itTrack->second);
    if (statId == m_selectedBuf)
    {
        return true;
    }

    return false;
}

bool AnselBuffer::isResourceTracked(HCLIENTRESOURCE resource) const
{
    // Check if any buffer instance was tracked with this buffer resource
    return m_resClearTracker.find(resource) != m_resClearTracker.end();
}

// Using the stats mechanism, decide which buffer is deemed most likely to have correct information from the
// buffers that we've captured statics on. In case we've already selected a buffer, clear tracking data such as
// times when buffers were bound/cleared, and delete any buffers that weren't bound this frame. If we did
// not see our selected buffer bound this frame, reset stats and start a new round of stats collection on the
// next frame.
void AnselBuffer::resolveStats()
{
    if (!useStats())
    {
        return;
    }

    // If user has selected a specific buffer using BufferTestingOptions filter, it will be selected
    // inside this function
    resolveStatsDebugSelectBuf();

    // Check if we've previously selected a buffer
    if (m_selectedBuf != BufferStats::s_NullId)
    {
        // If the selected buffer wasn't bound this frame, we'll not set it this time, and trigger a
        // stat recallwlation on the next frame.
        if (!isResourceTracked(BufferStats::getResFromId(m_selectedBuf)))
        {
            clearSelectedBuf();
        }

        clearPerFrameData();

        return;
    }

    // Make sure BufStatsOrdered is empty, since we're about to repopulate it based on the BufStats
    // which were collected during the frame that just happened
    m_BufStatsOrdered.clear();

    // Find the buffer which had the highest score. Meanwhile, erase bufs from database if they have a score of 0
    for (auto& itBuf = m_BufStats.begin(); itBuf != m_BufStats.end();)
    {
        const size_t lwrScore = callwlateScore(itBuf->second);
        if (lwrScore == 0)
        {
            itBuf = m_BufStats.erase(itBuf);
            continue;
        }

        // Make a copy of non-zero scoring buffers in BufStatsOrdered
        m_BufStatsOrdered.push_back(itBuf->second);

        ++itBuf;
    }

    // Order the stats and select our highest scored buffer. Any overrides from the BufferTestingOptions filter
    // will happen on the next frame
    if (!m_BufStatsOrdered.empty())
    {
        std::sort(m_BufStatsOrdered.begin(), m_BufStatsOrdered.end());
        std::reverse(m_BufStatsOrdered.begin(), m_BufStatsOrdered.end());

        m_selectedBuf = m_BufStatsOrdered[0].getId();
    }
    else
    {
        m_selectedBuf = BufferStats::s_NullId;
    }

    clearPerFrameData();
}

void AnselBuffer::resolveStatsDebugSelectBuf()
{
    if (!useStats())
    {
        return;
    }

    // Nothing to do if user didn't select a specific buffer from the BufferTestingOptions filter
    if (m_orderedBufSelect == 0)
    {
        return;
    }

    // If no buff has been selected, let a retrigger of stats occur rather than forcing a selection now
    if (m_selectedBuf == BufferStats::s_NullId)
    {
        return;
    }

    if (m_BufStatsOrdered.empty())
    {
        clearSelectedBuf();
        return;
    }

    // Select the buffer chosen from the filter
    const size_t bufSelect = min(m_orderedBufSelect, m_BufStatsOrdered.size());
    m_selectedBuf = m_BufStatsOrdered[bufSelect - 1].getId();

    LOG_BUFFER_EXTRA(DEBUG, "BufferTestingOptions Buffer #%llu selected, Id: 0x%p.%u", bufSelect, BufferStats::getResFromId(m_selectedBuf), BufferStats::getInstFromId(m_selectedBuf));
}

void AnselBuffer::addStats(HCLIENTRESOURCE resource, const AnselDeviceStates& devState)
{
    // Skip tracking stats if stats if disabled
    if (!useStats())
    {
        return;
    }

    // Draw how many time a draw call happened on this resource, per frame. Needs to always occur when
    // stats are being recorded, since Hudless copying works based on the number of draw calls
    m_BufStatsDrawCount[resource]++;

    // Skip tracking stats if we've already identified a buffer
    if (m_selectedBuf != BufferStats::s_NullId)
    {
        return;
    }

    // Only track stats for buffers that have been bound
    const auto& itTrack = m_resClearTracker.find(resource);
    if (itTrack == m_resClearTracker.cend())
    {
        return;
    }

    const BufferStats::Id statId = BufferStats::genId(resource, itTrack->second);

    auto& itBufStat = m_BufStats.find(statId);
    if (itBufStat == m_BufStats.end())
    {
        assert(!"Tried to add stats to a buffer that was never bound or copied!");
        return;
    }

    AnselResource* pData = m_pBufferInterface->lookupAnselResource(resource);
    if (pData == nullptr)
    {
        HCLIENTRESOURCE oldresource = m_clientResource;
        m_clientResource = resource;
        checkCreateServerGraphicsResources();
        m_clientResource = oldresource;

        pData = m_pBufferInterface->lookupAnselResource(resource);
    }

    // Only track buffers that have manageable formats
    if (pData != nullptr && shadermod::ir::ircolwert::DXGIformatToFormat(pData->toServerRes.format, true) != shadermod::ir::FragmentFormat::kUnknown)
    {
        itBufStat->second.update(devState, m_BufStatsDrawCount[resource]);
    }
}

void AnselBuffer::clearSelectedBuf()
{
    m_selectedBuf = BufferStats::s_NullId;
}

size_t AnselBuffer::getBufferSelect() const
{
    return m_orderedBufSelect;
}

void AnselBuffer::setBufferSelect(size_t buffer)
{
    m_orderedBufSelect = buffer;
}

size_t AnselBuffer::callwlateScore(BufferStats &stats) const
{
    assert(!"Virtual function callwlateScore was not overridden in this child class");
    return 0;
}

void AnselBuffer::clearPerFrameData()
{
    // Set the client resource to our selectedBuf, which might be a nullptr in the case that our
    // selected buf was not bound this frame. This will be the buffer used for resolving to the Present buffer
    setClientResource(BufferStats::getResFromId(m_selectedBuf));

    // Reset the buf tracker so we can check which buffers weren't touched on the next frame
    m_resClearTracker.clear();

    // Clear the draw calls counted per bound buffer
    m_BufStatsDrawCount.clear();

    if (m_selectedBuf == BufferStats::s_NullId)
    {
        // If no buffer was selected on a frame, we'll want to clear buffers in preparation
        // for for a recallwlation of stats. We don't clear the BufStats completely so we can avoid
        // reallocations
        for (auto& stat : m_BufStats)
        {
            stat.second.clear();
        }
    }
}

HRESULT AnselBufferDepth::selectBuffer(HCLIENTRESOURCE clientResource)
{
    if (getClientResource() != nullptr)
    {
        return S_OK;
    }

    return AnselBuffer::selectBuffer(clientResource);
}

HRESULT AnselBufferDepth::checkBuffer(HCLIENTRESOURCE clientResource, const AnselDeviceStates& deviceStates, DWORD width, DWORD height, bool* result) const
{
    assert(result);
    HRESULT status = S_OK;
    if (!result)
    {
        return E_FAIL;
    }
    *result = false;

    // 1. Check that we actually care to determine a depth buffer
    if (m_type != Type::kDepth || isForced())
    {
        return S_OK;
    }

    // 2. Validate input parameters
    if (clientResource == nullptr)
    {
        return S_OK;
    }
    
    // 3. Check that we haven't already picked a depth buffer
    if (m_didCopyFromClient || getClientResource() != nullptr)
    {
        return S_OK;
    }

    // 4. Check whether the device state indicates that the lwrrently bound buffer has depth info
    if (!isOpaqueDraw(deviceStates))
    {
        return S_OK;
    }

    // 5. Validate the resource object
    AnselClientResourceInfo resourceInfo;
    if (!SUCCEEDED(status = m_pBufferInterface->getClientResourceInfo(clientResource, &resourceInfo)))
    {
        LOG_BUFFER_EXTRA(ERROR, "Getting depth resource info failed");
        AnselBufferHandleFailure();
    }

    LOG_BUFFER_EXTRA(DEBUG, "Depth candidate: %dx%d:(format=%d)", resourceInfo.Width, resourceInfo.Height, resourceInfo.Format);

    // 6. Check if the resource dimensions directly match the present buffer dimensions
    bool resolutionsMatch = resourceInfo.Width == width && resourceInfo.Height == height;

#if (HEUR_RESIZED_BUFFERS == 1)
    if (!resolutionsMatch)
    {
        LOG_BUFFER_EXTRA(DEBUG, "Depth failure: resolution mismatch");

        const float aspectRatioTarget = (float)width / height;
        const float viewportRatio = deviceStates.ViewportWidth / deviceStates.ViewportHeight;
        const float backBuffRatio = (float)deviceStates.backbufferWidth / deviceStates.backbufferHeight;

        // This code is based on the Witness case, but might be applicable for other titles as well
        // Witness has internal HDR buffer that is bigger than the actual presentable RTV in the windowed mode
        const float aspectRatioRTV = (float)resourceInfo.Width / resourceInfo.Height;

        // TODO avoroshilov:
        // This is divider, as some games render to slightly lower res than the screen res
        // we need to determine that empirically later
        const float buffersScale = (float)resourceInfo.Width / width;

        // 7. Check if the resource dimensions directly match the screen dimensions
        if (std::fabs(aspectRatioTarget - aspectRatioRTV) < HEUR_RESIZED_BUFFERS_ASP_EPS && (HEUR_RESIZED_BUFFERS_SCALEMIN <= buffersScale) && (buffersScale <= HEUR_RESIZED_BUFFERS_SCALEMAX))
        {
            LOG_BUFFER_EXTRA(DEBUG, "Depth enhanced resolution check success: desiredAspectRatio=%f(%dx%d) : thisResourceAspectRatio=%f(%dx%d:(format=%d))", aspectRatioTarget, width, height, aspectRatioRTV, resourceInfo.Width, resourceInfo.Height, resourceInfo.Format);
            resolutionsMatch = true;
        }
        // 8. Check if resource viewport dim ratios align with the device backbuffer ratio dimensions
        else if (useViewportChecks() && std::fabs(backBuffRatio - viewportRatio) < HEUR_RESIZED_BUFFERS_ASP_EPS)
        {
            LOG_BUFFER_EXTRA(DEBUG, "Depth viewport check success: backBuffRatio=%f(%dx%d) : viewportRatio=%f(%fx%f)", backBuffRatio, deviceStates.backbufferWidth, deviceStates.backbufferHeight, viewportRatio, deviceStates.ViewportWidth, deviceStates.ViewportHeight);
            resolutionsMatch = true;
        }
        else
        {
            LOG_BUFFER_EXTRA(DEBUG, "Depth failure: enhanced resolution check failure: desiredAspectRatio=%f(%dx%d) : thisResourceAspectRatio=%f(%dx%d:(format=%d))", aspectRatioTarget, width, height, aspectRatioRTV, resourceInfo.Width, resourceInfo.Height, resourceInfo.Format);
        }
    }
#endif

    if (resolutionsMatch)
    {
        LOG_DEBUG(LogChannel::kRenderBuffer, "Depth buffer accept: %dx%d:(format=%d)", resourceInfo.Width, resourceInfo.Height, resourceInfo.Format);
        *result = true;
    }

    return S_OK;

}

void AnselBufferDepth::setViewportChecksEn(bool enable)
{
    m_useViewportChecks = enable;
}

bool AnselBufferDepth::useViewportChecks() const
{
    return m_useViewportChecks;
}

void AnselBufferDepth::setViewportScalingEn(bool enable)
{
    m_useViewportScaling = enable;
}

bool AnselBufferDepth::useViewportScaling() const
{
    return m_useViewportScaling;
}

void AnselBufferDepth::setWeights(const std::wstring& weightsStr)
{
    LOG_DEBUG(LogChannel::kRenderBuffer, "AnselBufferDepth received DRS Weight string: \"%ls\"", weightsStr.c_str());

    const std::vector<std::pair<std::wstring, uint32_t>> weights = parseWeightStr(weightsStr);
    for(const auto& weight : weights)
    {
        if (weight.first == L"Blend")
        {
            m_weightBlend = weight.second;
        }
        else if (weight.first == L"Stencil")
        {
            m_weightStencil = weight.second;
        }
        else if (weight.first == L"DepthOverBlend")
        {
            m_weightDepthOverBlend = weight.second;
        }
        else if (weight.first == L"DepthOverStencil")
        {
            m_weightDepthOverStencil = weight.second;
        }
        else if (weight.first == L"RtwColorMask")
        {
            m_weightRtwColorMask = weight.second;
        }
        else if (weight.first == L"VPEqualsBB")
        {
            m_weightVPEqualsBB = weight.second;
        }
        else if (weight.first == L"VPMatchesBBRatio")
        {
            m_weightVPMatchesBBRatio = weight.second;
        }
        else if (weight.first == L"VPTopZero")
        {
            m_weightVPTopZero = weight.second;
        }
        else
        {
            LOG_ERROR(LogChannel::kRenderBuffer, "AnselBufferDepth received unknown weight from DRS settings: %ls", weight.first.c_str());
            assert(!"AnselBufferDepth received unknown weight from DRS settings");
        }
    }
}

BufferStats::BufferStats(const Id& id, size_t bindNum)
: m_id(id)
, m_bindNum(bindNum)
{
    assert(bindNum != 0);
}

void BufferStats::update(const AnselDeviceStates& devState, size_t drawNum)
{
    // Don't update stats unless they've been cleared. This is so that we can retain stats information
    // when cycling through buffers for debugging
    if (m_score != 0)
    {
        return;
    }

    m_draws++;

    if (devState.DepthEnable)
    {
        m_depths++;
    }

    if (devState.BlendEnable)
    {
        m_blends++;
    }

    if (devState.StencilEnable)
    {
        m_stencils++;
    }

    m_rtwMasks[devState.RenderTargetWriteMask]++;
    m_backbufs[std::make_pair(devState.backbufferWidth, devState.backbufferHeight)]++;
    m_viewports[std::make_pair(devState.ViewportWidth, devState.ViewportHeight)]++;
    m_viewportTops[std::make_pair(devState.ViewportTopLeftX, devState.ViewportTopLeftY)]++;
}

void BufferStats::clear()
{
    m_score = 0;
    m_bindNum = 0;
    m_draws = 0;
    m_depths = 0;
    m_blends = 0;
    m_stencils = 0;

    m_cachedViewport = { 0.0f, 0.0f };

    m_rtwMasks.clear();
    m_backbufs.clear();
    m_viewports.clear();
    m_viewportTops.clear();
}

void BufferStats::processMappedStats()
{
    // Find the most common rtwMask value
    m_cachedRtwMask = std::max_element(m_rtwMasks.begin(), m_rtwMasks.end())->first;

    // Validate that we only ever see 1 backbuffer size
    assert(m_backbufs.size() == 1);
    m_cachedBackbuf = std::make_pair(
        static_cast<FLOAT>(m_backbufs.begin()->first.first),
        static_cast<FLOAT>(m_backbufs.begin()->first.second));

    // Find most common viewport, and erase any viewports that weren't seen this frame
    auto& itMostCommolwiewport = m_viewports.cend();
    if (m_viewports.size() == 1)
    {
        itMostCommolwiewport = m_viewports.cbegin();
    }
    else
    {
        for (auto& itBuf = m_viewports.cbegin(); itBuf != m_viewports.cend();)
        {
            if (itBuf->second == 0)
            {
                // Remove any viewports that haven't been seen recently
                itBuf = m_viewports.erase(itBuf);
            }
            else
            {
                if (itMostCommolwiewport == m_viewports.cend() || itBuf->second > itMostCommolwiewport->second)
                {
                    itMostCommolwiewport = itBuf;
                }
                ++itBuf;
            }
        }
    }
    m_cachedViewport = itMostCommolwiewport->first;
    
    // Find most common viewport top x/y, and erase any that weren't seen this frame
    auto& itMostCommolwiewportTop = m_viewportTops.cend();
    if (m_viewportTops.size() == 1)
    {
        itMostCommolwiewportTop = m_viewportTops.cbegin();
    }
    else
    {
        for (auto& itBuf = m_viewportTops.cbegin(); itBuf != m_viewportTops.cend();)
        {
            if (itBuf->second == 0)
            {
                // Remove any viewport tops that haven't been seen recently
                itBuf = m_viewportTops.erase(itBuf);
            }
            else
            {
                if (itMostCommolwiewportTop == m_viewportTops.cend() || itBuf->second > itMostCommolwiewportTop->second)
                {
                    itMostCommolwiewportTop = itBuf;
                }
                ++itBuf;
            }
        }
    }
    m_cachedViewportTop = itMostCommolwiewportTop->first;
}

size_t BufferStats::callwlateVPScore(size_t weightEqual, size_t weightRatio) const
{
    size_t vpScore = 0;
    if (getViewport() == getBackbuf())
    {
        vpScore = weightEqual;
    }
    else
    {
        const float backBuffRatio = getBackbuf().first / getBackbuf().second;
        const float viewportRatio = getViewport().first / getViewport().second;

        if (anselutils::areAlmostEqual(backBuffRatio, viewportRatio))
        {
            vpScore = weightRatio;
        }
    }

    return vpScore;
}

size_t AnselBufferDepth::callwlateScore(BufferStats &stats) const
{
    assert(stats.getId().first);

    // If no calls with depth happened, then clearly this is not a depth buffer
    if (stats.getDepths() == 0)
    {
        LOG_BUFFER_EXTRA(DEBUG, "Depth Scoring [0x%p.%u]: Didn't have any depths",  stats.getIdResource(), stats.getIdInstance());
        return 0;
    }

    // If we've already callwlated a score, return the cached value
    if (stats.getScore() != 0)
    {
        return stats.getScore();
    }

    LOG_BUFFER_EXTRA(DEBUG, "** Score Data for Depth Buffer: 0x%p.%u",  stats.getIdResource(), stats.getIdInstance());
    LOG_BUFFER_EXTRA(DEBUG, "- Num Depths: %llu", stats.getDepths());

    // More draw calls suggests this is a depth buffer
    const size_t drawScore = static_cast<size_t>(log2(stats.getDraws()));
    LOG_BUFFER_EXTRA(DEBUG, "- Num Draws: %llu (%llu)", stats.getDraws(), drawScore);
    
    // Having a mix of blend and stencil draws suggests a depth buffer
    const size_t blendScore = stats.getBlends() > 0 ? m_weightBlend : 0;
    LOG_BUFFER_EXTRA(DEBUG, "- Num Blends: %llu (%llu)", stats.getBlends(), blendScore);

    const size_t stencilScore = stats.getStencils() > 0 ? m_weightStencil : 0;
    LOG_BUFFER_EXTRA(DEBUG, "- Num Stencils: %llu (%llu)", stats.getStencils(), stencilScore);

    // Having a majority of depth calls suggests a depth buffer
    const size_t depthOverBlendScore = stats.getDepths() >= stats.getBlends() ? m_weightDepthOverBlend : 0;
    LOG_BUFFER_EXTRA(DEBUG, "- Depths over blend (%llu)", depthOverBlendScore);

    const size_t depthOverStencilScore = stats.getDepths() >= stats.getStencils() ? m_weightDepthOverStencil : 0;
    LOG_BUFFER_EXTRA(DEBUG, "- Depths over stencil (%llu)", depthOverStencilScore);

    // Gather information some of the stats
    stats.processMappedStats();

    // Check if most common RenderTargetWriteMask is set to write all
    const size_t rtwColorMaskScore = stats.getRtwMask() == D3D11_COLOR_WRITE_ENABLE_ALL ? m_weightRtwColorMask : 0;
    LOG_BUFFER_EXTRA(DEBUG, "- Most common RTW Mask: 0x%hu (%llu)", stats.getRtwMask(), rtwColorMaskScore);

    // Check if most common viewport is equal or aligned to backbuf
    size_t commolwPScore = stats.callwlateVPScore(m_weightVPEqualsBB, m_weightVPMatchesBBRatio);
    LOG_BUFFER_EXTRA(DEBUG, "- Most common viewport: %.02f,%.02f backbuf: %.02f,%.02f (%llu)",
        stats.getViewport().first, stats.getViewport().second, stats.getBackbuf().first, stats.getBackbuf().second, commolwPScore);

    // Check if most common viewport top is equal to (0, 0)
    const size_t commolwPTopScore = stats.getViewportTop() == std::make_pair(0.0f, 0.0f) ? m_weightVPTopZero : 0;
    LOG_BUFFER_EXTRA(DEBUG, "- Most common viewport top %.02f,%.02f (%llu)",
        stats.getViewportTop().first, stats.getViewportTop().second, commolwPTopScore);

    // Tally the final score
    stats.setScore(drawScore +
                blendScore +
                stencilScore +
                depthOverBlendScore +
                depthOverStencilScore +
                rtwColorMaskScore +
                commolwPScore +
                commolwPTopScore);

    LOG_BUFFER_EXTRA(DEBUG, "--- Final Score: %llu", stats.getScore());

    return stats.getScore();
}

bool AnselBufferDepth::isOpaqueDraw(const AnselDeviceStates &deviceStates)
{
    HRESULT status = S_OK;
    if (deviceStates.DepthEnable == FALSE)
    {
        return false;
    }

    if (deviceStates.BlendEnable == TRUE)
    {
        return false;
    }

    LOG_BUFFER_EXTRA(DEBUG, "Draw opaque");

    return true;
}

std::set<DXGI_FORMAT> AnselBufferHDR::s_supportedFormats = {
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    DXGI_FORMAT_R32G32B32A32_TYPELESS,
    DXGI_FORMAT_R32G32B32_FLOAT,
    DXGI_FORMAT_R32G32B32_TYPELESS,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
    DXGI_FORMAT_R16G16B16A16_TYPELESS,
    DXGI_FORMAT_R11G11B10_FLOAT
};

bool AnselBufferHDR::isFormatSupported(DWORD format)
{
    return s_supportedFormats.find(static_cast<DXGI_FORMAT>(format)) != s_supportedFormats.cend();
}

HRESULT AnselBufferHDR::checkBuffer(HCLIENTRESOURCE clientResource, DWORD width, DWORD height, bool* result)
{
    assert(result);

    HRESULT status = S_OK;
    if (!result)
        return E_FAIL;
    *result = false;

    AnselClientResourceInfo resourceInfo;
    if (!SUCCEEDED(status = m_pBufferInterface->getClientResourceInfo(clientResource, &resourceInfo)))
    {
        LOG_BUFFER_EXTRA(ERROR, "Getting HDR resource info failed");
        AnselBufferHandleFailure();
    }

    // basic heuristic - in case the format is one of HDR bitness and dimensions are the same - consider it HDR
    if (!isForced() && isFormatSupported(resourceInfo.Format))
    {
        LOG_BUFFER_EXTRA(DEBUG, "HDR candidate: %dx%d:(format=%d)", resourceInfo.Width, resourceInfo.Height, resourceInfo.Format);

        bool resolutionsMatch = resourceInfo.Width == width && resourceInfo.Height == height;
#if (HEUR_RESIZED_BUFFERS == 1)
        if (!resolutionsMatch)
        {
            // This code is based on the Witness case, but might be applicable for other titles as well
            // Witness has internal HDR buffer that is bigger than the actual presentable RTV in the windowed mode
            float aspectRatioTarget = (float)width / height;
            float aspectRatioRTV = (float)resourceInfo.Width / resourceInfo.Height;

            LOG_BUFFER_EXTRA(DEBUG, "HDR failure: resolution mismatch");

            // TODO avoroshilov:
            // This is divider, as some games render to slightly lower res than the screen res
            // we need to determine that empirically later
            const float buffersScale = (float)resourceInfo.Width / width;
            if (std::abs(aspectRatioTarget - aspectRatioRTV) < HEUR_RESIZED_BUFFERS_ASP_EPS && (HEUR_RESIZED_BUFFERS_SCALEMIN <= buffersScale) && (buffersScale <= HEUR_RESIZED_BUFFERS_SCALEMAX))
            {
                LOG_DEBUG(LogChannel::kRenderBuffer, "HDR enhanced resolution check success: desiredAspectRatio=%f(%dx%d) : thisResourceAspectRatio=%f(%dx%d:(format=%d))", aspectRatioTarget, width, height, aspectRatioRTV, resourceInfo.Width, resourceInfo.Height, resourceInfo.Format);
                resolutionsMatch = true;
            }
            else
            {
                LOG_BUFFER_EXTRA(DEBUG, "HDR failure: enhanced resolution check failure: desiredAspectRatio=%f(%dx%d) : thisResourceAspectRatio=%f(%dx%d:(format=%d))", aspectRatioTarget, width, height, aspectRatioRTV, resourceInfo.Width, resourceInfo.Height, resourceInfo.Format);
            }
        }
#endif
        if (resolutionsMatch)
        {
            LOG_DEBUG(LogChannel::kRenderBuffer, "HDR buffer accept: %dx%d:(format=%d)", resourceInfo.Width, resourceInfo.Height, resourceInfo.Format);
            *result = true;
            return S_OK;
        }
    }
    return S_OK;

}

HRESULT AnselBufferHudless::checkBuffer(HCLIENTRESOURCE clientResource, const AnselDeviceStates& deviceStates, bool* result)
{
    assert(result);

    HRESULT status = S_OK;
    if (!result)
    {
        return E_FAIL;
    }
    *result = false;

    if (isForced())
    {
        return S_OK;
    }

    if (!m_didCopyFromClient && clientResource && isHUDlessColor(deviceStates))
    {
        LOG_DEBUG(LogChannel::kRenderBuffer, "HUDless buffer accept");
        *result = true;
    }
    return S_OK;
}

bool AnselBufferHudless::copyHudless(HCLIENTRESOURCE resource)
{
    // Skip this check if stats is disabled
    if (!useStats())
    {
        return false;
    }

    // Only indicate we want to copy if this buffer matches our selected buffer
    if (!compareAgainstSelBuf(resource))
    {
        return false;
    }

    // Validate the resource has been previously bound
    assert(m_resClearTracker.find(resource) != m_resClearTracker.end());

    // If m_compareDrawNum is zero, it means we want to copy after the last draw call, which will happen
    // in finalizeFrame(), so don't copy now
    if (m_compareDrawNum == 0 || m_compareDrawNum != m_BufStatsDrawCount[resource])
    {
        return false;
    }

    // Perform the copy now
    selectBuffer(resource);
    copyResource(0);

    return true;
}

 size_t AnselBufferHudless::getCompareDrawNum() const
{
    return m_compareDrawNum;
}

void AnselBufferHudless::setCompareDrawNum(size_t compareDrawNum)
{
    m_compareDrawNum = compareDrawNum;
}

void AnselBufferHudless::setSingleRTV(bool enable)
{
    m_useSingleRTV = enable;
}

bool AnselBufferHudless::useSingleRTV() const
{
    return m_useSingleRTV;
}

void AnselBufferHudless::setRestrictFormats(bool enable)
{
    m_restrictFormats = enable;
}

void AnselBufferHudless::setWeights(const std::wstring& weightsStr)
{
    LOG_DEBUG(LogChannel::kRenderBuffer, "AnselBufferHudless received DRS Weight string: \"%ls\"", weightsStr.c_str());

    const std::vector<std::pair<std::wstring, uint32_t>> weights = parseWeightStr(weightsStr);
    for(const auto& weight : weights)
    {
        if (weight.first == L"Bind")
        {
            m_weightBind = weight.second;
        }
        else if (weight.first == L"DrawDepthDiff")
        {
            m_weightDrawDepthDiff = weight.second;
        }
        else if (weight.first == L"ZeroStencil")
        {
            m_weightZeroStencil = weight.second;
        }
        else if (weight.first == L"VPEqualsBB")
        {
            m_weightVPEqualsBB = weight.second;
        }
        else if (weight.first == L"VPMatchesBBRatio")
        {
            m_weightVPMatchesBBRatio = weight.second;
        }
        else if (weight.first == L"SingleViewport")
        {
            m_weightSingleViewport = weight.second;
        }
        else
        {
            LOG_ERROR(LogChannel::kRenderBuffer, "AnselBufferDepth received unknown weight from DRS settings: %ls", weight.first.c_str());
            assert(!"AnselBufferDepth received unknown weight from DRS settings");
        }
    }
}

bool AnselBufferHudless::useRestrictFormats() const
{
    return m_restrictFormats;
}

const std::set<DXGI_FORMAT> AnselBufferHudless::s_supportedFormats = {
    DXGI_FORMAT_R8G8B8A8_UNORM,         // Witness, Fallout 4
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,    // Sponza
    DXGI_FORMAT_R10G10B10A2_UNORM,      // Borderlands
    DXGI_FORMAT_R11G11B10_FLOAT         // MWH
};

bool AnselBufferHudless::isFormatSupported(DWORD format)
{
    const bool isFormatKnown = shadermod::ir::ircolwert::DXGIformatToFormat(static_cast<DXGI_FORMAT>(format), true) !=
        shadermod::ir::FragmentFormat::kUnknown;

    const bool isFormatSupported = s_supportedFormats.find(static_cast<DXGI_FORMAT>(format)) != s_supportedFormats.cend();

    return isFormatKnown && isFormatSupported;
}

size_t AnselBufferHudless::callwlateScore(BufferStats& stats) const
{
    assert(stats.getId() != BufferStats::s_NullId);

    if (stats.getDraws() == 0)
    {
        LOG_BUFFER_EXTRA(DEBUG, "Hudless Scoring [0x%p.%u]: Didn't have any draws", stats.getIdResource(), stats.getIdInstance());
        return 0;
    }

    // If we've already callwlated a score, return the cached value
    if (stats.getScore() != 0)
    {
        return stats.getScore();
    }

    LOG_BUFFER_EXTRA(DEBUG, "** Score Data for Hudless Buffer: 0x%p.%u", stats.getIdResource(), stats.getIdInstance());

    // Hud buffer is likely the last buffer seen
    const size_t bindScore = stats.getBindNum() * m_weightBind;
    LOG_BUFFER_EXTRA(DEBUG, "- Bind Count: %llu (%llu)", stats.getBindNum(), bindScore);

    // Hud buffer is more likely to have a difference between number of Draw calls vs Depth
    const size_t drawDepthDiffScore = (stats.getDraws() - stats.getDepths()) * m_weightDrawDepthDiff;
    LOG_BUFFER_EXTRA(DEBUG, "- Draw vs Depth: %llu, %llu (%llu)", stats.getDraws(), stats.getDepths(), drawDepthDiffScore);

    // Hud buffer is more likely to have no stencil draw calls
    const size_t zeroStencilScore = (stats.getStencils() > 0 ? m_weightZeroStencil : 0);
    LOG_BUFFER_EXTRA(DEBUG, "- Stencil count: %llu (%llu)", stats.getStencils(), zeroStencilScore);

    // Gather information some of the stats
    stats.processMappedStats();

    // Check if most common viewport is equal or aligned to backbuf
    size_t commolwPScore = stats.callwlateVPScore(m_weightVPEqualsBB, m_weightVPMatchesBBRatio);
    LOG_BUFFER_EXTRA(DEBUG, "- Most common viewport: %.02f,%.02f backbuf: %.02f,%.02f (%llu)",
        stats.getViewport().first, stats.getViewport().second, stats.getBackbuf().first, stats.getBackbuf().second, commolwPScore);

    const size_t singleVPScore = stats.getViewportsSize() == 1 ? m_weightSingleViewport : 0;
    LOG_BUFFER_EXTRA(DEBUG, "- Num Viewports: %llu (%llu)", stats.getViewportsSize(), singleVPScore);

    // Tally the final score
    stats.setScore(bindScore +
        drawDepthDiffScore +
        zeroStencilScore +
        commolwPScore +
        singleVPScore +
        1); // Plus 1 added in case all other parameters are zero

    LOG_BUFFER_EXTRA(DEBUG, "--- Final Score: %llu", stats.getScore());

    return stats.getScore();
}

void AnselBuffer::destroy()
{
    SAFE_RELEASE(m_resourceCopy.pRTV);
    SAFE_RELEASE(m_resourceCopy.pSRV);
    SAFE_RELEASE(m_resourceCopy.pTexture2D);
}

AnselBufferDB::~AnselBufferDB()
{
    for (size_t i = 0; i < static_cast<size_t>(AnselBuffer::Type::kNUM_ENTRIES); i++)
    {
        m_bufs[i]->release();
        m_bufs[i]->destroy();

        delete m_bufs[i];
        m_bufs[i] = nullptr;
    }
}

AnselBufferPresent& AnselBufferDB::Present() const
{
    return *static_cast<AnselBufferPresent*>(GetBuf(AnselBuffer::Type::kPresent));
}

AnselBufferFinal& AnselBufferDB::Final() const
{
    return *static_cast<AnselBufferFinal*>(GetBuf(AnselBuffer::Type::kFinal));
}

AnselBufferDepth& AnselBufferDB::Depth() const
{
    return *static_cast<AnselBufferDepth*>(GetBuf(AnselBuffer::Type::kDepth));
}

AnselBufferHDR& AnselBufferDB::HDR() const
{
    return *static_cast<AnselBufferHDR*>(GetBuf(AnselBuffer::Type::kHDR));
}

AnselBufferHudless& AnselBufferDB::Hudless() const
{
    return *static_cast<AnselBufferHudless*>(GetBuf(AnselBuffer::Type::kHudless));
}

AnselBuffer* AnselBufferDB::GetBuf(AnselBuffer::Type type) const
{
    assert(type < AnselBuffer::Type::kNUM_ENTRIES);
    return m_bufs[static_cast<size_t>(type)];
}

bool AnselBufferDB::setBuffersInterfaceIfNeeded(AnselBufferInterface * pBufferInterface)
{
    if (m_bufs[0]->m_pBufferInterface != pBufferInterface)
    {
        for (auto &pBuf : m_bufs)
        {
            pBuf->m_pBufferInterface = pBufferInterface;
        }
        return true;
    }

    return false;
}

bool AnselBufferDB::clearAnselResource(const AnselResource* pAnselResource)
{
    bool found = false;
    for (auto &pBuf : m_bufs)
    {
        if (pBuf->getAnselResource() == pAnselResource)
        {
            pBuf->setAnselResource(nullptr);
            found = true;
        }
    }

    return found;
}

bool AnselBufferDB::clearClientResource(HCLIENTRESOURCE clientResource)
{
    bool found = false;
    for (auto &pBuf : m_bufs)
    {
        if (pBuf->getClientResource() == clientResource)
        {
            pBuf->setClientResource(nullptr);
            found = true;
        }
    }
    return found;
}

void AnselBufferDB::release()
{
    for (auto &pBuf : m_bufs)
    {
        pBuf->release();
    }
}

void AnselBufferDB::destroy()
{
    for (auto &pBuf : m_bufs)
    {
        pBuf->destroy();
    }
}

#undef AnselBufferHandleFailure
#undef SAFE_RELEASE
#undef LOG_BUFFER_EXTRA
#undef LOG_BUFFER_EXTRA_ENABLED
