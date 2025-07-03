#include <d3d9.h>
#include <d3d9types.h>
#include <dxva2api.h>
#include "dxvahdapi.h"

HRESULT IDXVAHD_VideoProcessor::SetVideoProcessBltState(
	DXVAHD_BLT_STATE State,
	UINT             DataSize,
	VOID*            pData
)
{
	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::GetVideoProcessBltState(
	DXVAHD_BLT_STATE State,
	UINT             DataSize,
	VOID*            pData
)
{
	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::SetVideoProcessSlotState(
	UINT              SlotNumber,
	DXVAHD_SLOT_STATE State,
	UINT              DataSize,
	VOID*             pData
)
{
	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::GetVideoProcessSlotState(
	UINT              SlotNumber,
	DXVAHD_SLOT_STATE State,
	UINT              DataSize,
	VOID*             pData
)
{
	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::VideoProcessBltHD(
	IDirect3DSurface9* pOutputSurface,
	UINT               OutputFrame,
	UINT               SlotCount,
	DXVAHD_SLOT_DATA*  pData
)
{	// to be complete
	int i;
#if 1
	for (i = 0; i < 7; i++)
	{
		//InputVideoSamples[i].Start = pData[i].InputVideoSamples.Start;
		//InputVideoSamples[i].End = pData[i].InputVideoSamples.End;
		InputVideoSamples[i].SampleFormat = pData[i].InputVideoSamples.SampleFormat;
		InputVideoSamples[i].SrcSurface = pData[i].InputVideoSamples.SrcSurface;
		InputVideoSamples[i].SampleData = pData[i].InputVideoSamples.SampleData;
		InputVideoSamples[i].PlanarAlpha = pData[i].InputVideoSamples.PlanarAlpha;
		
		SetRect(&InputVideoSamples[i].SrcRect,
			pData[i].InputVideoSamples.SrcRect.left,
			pData[i].InputVideoSamples.SrcRect.top,
			pData[i].InputVideoSamples.SrcRect.right,
			pData[i].InputVideoSamples.SrcRect.bottom);

		SetRect(&InputVideoSamples[i].DstRect,
			pData[i].InputVideoSamples.DstRect.left,
			pData[i].InputVideoSamples.DstRect.top,
			pData[i].InputVideoSamples.DstRect.right,
			pData[i].InputVideoSamples.DstRect.bottom);
	}
#endif

	HRESULT hr = m_pVideodProcessDevice->VideoProcessBlt(
			pOutputSurface,
			&VideoProcessBltParams,
			&InputVideoSamples[0],
			SlotCount,
			NULL);  // Reserved, must be NULL

	return hr;
}

HRESULT IDXVAHD_Device::GetVideoProcessorOutputFormatCount(
	UINT* pCount
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::GetVideoProcessorOutputFormats(
	UINT Count,
	D3DFORMAT* pFormat
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::GetVideoProcessorInputFormatCount(
	UINT* pCount
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::GetVideoProcessorInputFormats(
	UINT Count,
	D3DFORMAT* pFormat
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::CreateVideoSurface(
	UINT                Width,
	UINT                Height,
	D3DFORMAT           Format,
	UINT                NumSurfaces,
	D3DPOOL             Pool,
	DXVAHD_SURFACE_TYPE Type,
	IDirect3DSurface9** ppSurface
)
{
    HRESULT hr;
	hr = m_pAccelServices->CreateSurface(
		Width,
		Height, 
		0,									// no back buffers
		Format,								// YUY2 format (4:2:2)
		D3DPOOL_DEFAULT,					// Default pool.
		0,									// Reserved, use zero.
		DXVA2_VideoProcessorRenderTarget,	// Create a video processor render target surface.
		ppSurface,
		NULL								// not shared
	);
	return hr;
}

HRESULT IDXVAHD_Device::GetVideoProcessorDeviceCaps(
	DXVAHD_VPDEVCAPS* pCaps
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::GetVideoProcessorFilterRange(
	DXVAHD_FILTER_RANGE       Filter,
	DXVAHD_FILTER_RANGE_DATA* pData
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::GetVideoProcessorCapsCount(
	UINT* pCount
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::GetVideoProcessorCaps(
	UINT Count,
	DXVAHD_VPCAPS* pCaps
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::GetVideoProcessorFrameRateCount(
	GUID  VPGuid,
	UINT* pCount
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::GetVideoProcessorFrameRates(
	GUID               VPGuid,
	UINT               Count,
	DXVAHD_FRAME_RATE* pFrameRate
)
{
	return S_OK;
}

HRESULT IDXVAHD_Device::CreateVideoProcessor(
	GUID                  VPGuid,
	IDXVAHD_VideoProcessor** ppVideoProcessor
)
{
    HRESULT hr;
	IDXVAHD_VideoProcessor* pVideoProcessor = new(IDXVAHD_VideoProcessor);
	
	// to borrow VideoProcessBlt()
	DXVA2_ExtendedFormat SampleFormat;
    DXVA2_VideoDesc videoDesc;    // Description of the video stream to process.

	SampleFormat.SampleFormat = DXVA2_SampleProgressiveFrame;
	SampleFormat.NominalRange = DXVA2_NominalRange_16_235;
	SampleFormat.VideoTransferMatrix = DXVA2_VideoTransferMatrix_BT601;
    videoDesc.SampleWidth  = 1920;
    videoDesc.SampleHeight = 1080,
    videoDesc.SampleFormat = SampleFormat;
    videoDesc.Format = (D3DFORMAT)'2YUY';
    videoDesc.InputSampleFreq.Numerator = 25;
    videoDesc.InputSampleFreq.Denominator = 1;
    videoDesc.OutputFrameFreq.Numerator = 25;
    videoDesc.OutputFrameFreq.Denominator = 1;

	hr = m_pAccelServices->CreateVideoProcessor(
		VPGuid,					// device to create
		&videoDesc,				// description of source video
		D3DFMT_X8R8G8B8,		// render target pixel format
		10,						// 4 substreams
		&(pVideoProcessor->m_pVideodProcessDevice));

	*ppVideoProcessor = pVideoProcessor;

	return S_OK;
}

HRESULT DXVAHD_CreateDevice(
    IDirect3DDevice9Ex*  pD3DDevice,
    DXVAHD_CONTENT_DESC* pContentDesc,
    DXVAHD_DEVICE_FLAG   Flags,
    HMODULE              VideoProcessorPlugin,
	IDXVAHD_Device**     ppDevice
)
{
    HRESULT hr;
	IDXVAHD_Device* pDevice = new(IDXVAHD_Device);
	
	hr = DXVA2CreateVideoService(pD3DDevice, IID_IDirectXVideoProcessorService,
                                     (void**)&(pDevice->m_pAccelServices));
	if (hr != S_OK)
		return hr;
	*ppDevice = pDevice;
	return S_OK;
}

