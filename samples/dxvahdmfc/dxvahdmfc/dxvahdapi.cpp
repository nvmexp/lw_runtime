#include <d3d9.h>
#include <d3d9types.h>
#include "dxva2api.h"
#include "dxvahdapi.h"

HRESULT IDXVAHD_VideoProcessor::OnCreate()
{
	int i, j;

	// default value for blt_state
	BltState.TargetRect.Enable = FALSE;
	SetRect(&BltState.TargetRect.TargetRect, 0, 0, 0, 0);

	BltState.BackgroundColor.YCbCr =FALSE;
	BltState.BackgroundColor.BackgroundColor.R = 0.;
	BltState.BackgroundColor.BackgroundColor.G = 0.;
	BltState.BackgroundColor.BackgroundColor.B = 0.;
	BltState.BackgroundColor.BackgroundColor.A = 1.;

	BltState.OutputColorSpace.Usage = 0;
	BltState.OutputColorSpace.RGB_Range = 0;
	BltState.OutputColorSpace.YCbCr_Matrix = 0;
	BltState.OutputColorSpace.YCbCr_xvYCC = 0;

	BltState.AlphaFill.Mode = DXVAHD_ALPHA_FILL_MODE_BACKGROUND;
	BltState.AlphaFill.StreamNumber = 0;

	BltState.DownSample.Enable = FALSE;
	BltState.DownSample.Size.cx = 1;
	BltState.DownSample.Size.cy = 1;

	BltState.ClearRect.Enable = FALSE;
	for (i = 0; i < 32; i++)
	{
		SetRect(&(BltState.ClearRect.ClearRect[i]), 0, 0, 0, 0);
	}

	// default values for stream_state
	for (i = 0; i < DXVAHD_MAX_NUM_STREAMS; i++)
	{
		pStreamState[i].D3DFormat.Format = D3DFMT_UNKNOWN;

		pStreamState[i].FrameFormat.FrameFormat = DXVAHD_FRAME_FORMAT_PROGRESSIVE;

		pStreamState[i].InputColorSpace.RGB_Range = 0;
		pStreamState[i].InputColorSpace.YCbCr_Matrix =0;
		pStreamState[i].InputColorSpace.YCbCr_xvYCC = 0;

		pStreamState[i].OutputRate.RepeatFrame = FALSE;
		pStreamState[i].OutputRate.OutputRate = DXVAHD_OUTPUT_RATE_NORMAL;
		pStreamState[i].OutputRate.LwstomRate.Denominator = 1;
		pStreamState[i].OutputRate.LwstomRate.Numerator = 1;

		pStreamState[i].SourceRect.Enable = FALSE;
		SetRect(&pStreamState[i].SourceRect.SourceRect, 0, 0, 0, 0);

		pStreamState[i].DestinationRect.Enable = FALSE;
		SetRect(&pStreamState[i].DestinationRect.DestinationRect, 0, 0, 0, 0);

		pStreamState[i].Alpha.Enable = FALSE;
		pStreamState[i].Alpha.Alpha = 1.0;
#if 0
		for (j = 0; j < 256; j++)
			pStreamState[i].Palette.Palette[j] = D3DCOLOR_ARGB(255, 255, 255, 255);
#endif
		pStreamState[i].ClearRect.ClearRectMask = 0;

		pStreamState[i].LumaKey.Enable = FALSE;
		pStreamState[i].LumaKey.Lower = 0.0;
		pStreamState[i].LumaKey.Upper = 0.0;

		pStreamState[i].Brightness.Enable = FALSE;
		pStreamState[i].Brightness.Level = 0;

		pStreamState[i].Contrast.Enable = FALSE;
		pStreamState[i].Contrast.Level = 0;

		pStreamState[i].Hue.Enable = FALSE;
		pStreamState[i].Hue.Level = 0;

		pStreamState[i].Saturation.Enable = FALSE;
		pStreamState[i].Saturation.Level = 0;

		pStreamState[i].NoiseReduction.Enable = FALSE;
		pStreamState[i].NoiseReduction.Level = 0;

		pStreamState[i].EdgeEnhancement.Enable = FALSE;
		pStreamState[i].EdgeEnhancement.Level = 0;

		pStreamState[i].AnamorphicScaling.Enable = FALSE;
		pStreamState[i].AnamorphicScaling.Level = 0;
	}

	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::SetVideoProcessBltState(
	DXVAHD_BLT_STATE State,
	UINT             DataSize,
	VOID*            pData
)
{
	switch (State)
	{
	case DXVAHD_BLT_STATE_TARGET_RECT:
		BltState.TargetRect = *((DXVAHD_BLT_STATE_TARGET_RECT_DATA *)pData);;
		break;
	case DXVAHD_BLT_STATE_BACKGROUND_COLOR:
		BltState.BackgroundColor = *((DXVAHD_BLT_STATE_BACKGROUND_COLOR_DATA *)pData);
		break;
	case DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE:
		BltState.OutputColorSpace = *((DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE_DATA *)pData);
		break;
	case DXVAHD_BLT_STATE_ALPHA_FILL:
		BltState.AlphaFill = *((DXVAHD_BLT_STATE_ALPHA_FILL_DATA *)pData);
		break;
	case DXVAHD_BLT_STATE_DOWNSAMPLE:
		BltState.DownSample = *((DXVAHD_BLT_STATE_DOWNSAMPLE_DATA *)pData);
		break;
	case DXVAHD_BLT_STATE_CLEAR_RECT:
		BltState.ClearRect = *((DXVAHD_BLT_STATE_CLEAR_RECT_DATA *)pData);
		break;
	}
	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::GetVideoProcessBltState(
	DXVAHD_BLT_STATE State,
	UINT             DataSize,
	VOID*            pData
)
{
	switch (State)
	{
	case DXVAHD_BLT_STATE_TARGET_RECT:
		*((DXVAHD_BLT_STATE_TARGET_RECT_DATA *)pData) = BltState.TargetRect;
		break;
	case DXVAHD_BLT_STATE_BACKGROUND_COLOR:
		*((DXVAHD_BLT_STATE_BACKGROUND_COLOR_DATA *)pData) = BltState.BackgroundColor;
		break;
	case DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE:
		*((DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE_DATA *)pData) = BltState.OutputColorSpace;
		break;
	case DXVAHD_BLT_STATE_ALPHA_FILL:
		*((DXVAHD_BLT_STATE_ALPHA_FILL_DATA *)pData) = BltState.AlphaFill;
		break;
	case DXVAHD_BLT_STATE_DOWNSAMPLE:
		*((DXVAHD_BLT_STATE_DOWNSAMPLE_DATA *)pData) = BltState.DownSample;
		break;
	case DXVAHD_BLT_STATE_CLEAR_RECT:
		*((DXVAHD_BLT_STATE_CLEAR_RECT_DATA *)pData)= BltState.ClearRect;
		break;
	}
	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::SetVideoProcessStreamState(
	UINT              StreamNumber,
	DXVAHD_STREAM_STATE State,
	UINT              DataSize,
	VOID*             pData
)
{
	switch (State)
	{
	case DXVAHD_STREAM_STATE_D3DFORMAT:
		pStreamState[StreamNumber].D3DFormat = *((DXVAHD_STREAM_STATE_D3DFORMAT_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_FRAME_FORMAT:
		pStreamState[StreamNumber].FrameFormat = *((DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE:
		pStreamState[StreamNumber].InputColorSpace = *((DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_OUTPUT_RATE:
		pStreamState[StreamNumber].OutputRate = *((DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_SOURCE_RECT:
		pStreamState[StreamNumber].SourceRect = *((DXVAHD_STREAM_STATE_SOURCE_RECT_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_DESTINATION_RECT:
		pStreamState[StreamNumber].DestinationRect = *((DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_ALPHA:
		pStreamState[StreamNumber].Alpha = *((DXVAHD_STREAM_STATE_ALPHA_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_PALETTE:
		pStreamState[StreamNumber].Palette = *((DXVAHD_STREAM_STATE_PALETTE_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_CLEAR_RECT:
		pStreamState[StreamNumber].ClearRect = *((DXVAHD_STREAM_STATE_CLEAR_RECT_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_LUMA_KEY:
		pStreamState[StreamNumber].LumaKey = *((DXVAHD_STREAM_STATE_LUMA_KEY_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS:
		pStreamState[StreamNumber].Brightness = *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_FILTER_CONTRAST:
		pStreamState[StreamNumber].Contrast = *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_FILTER_HUE:
		pStreamState[StreamNumber].Hue = *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_FILTER_SATURATION:
		pStreamState[StreamNumber].Saturation = *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION:
		pStreamState[StreamNumber].NoiseReduction = *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT:
		pStreamState[StreamNumber].EdgeEnhancement = *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData);
		break;
	case DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING:
		pStreamState[StreamNumber].AnamorphicScaling = *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData);
		break;
	}
	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::GetVideoProcessStreamState(
	UINT              StreamNumber,
	DXVAHD_STREAM_STATE State,
	UINT              DataSize,
	VOID*             pData
)
{
	switch (State)
	{
	case DXVAHD_STREAM_STATE_D3DFORMAT:
		 *((DXVAHD_STREAM_STATE_D3DFORMAT_DATA *)pData) = pStreamState[StreamNumber].D3DFormat;
		break;
	case DXVAHD_STREAM_STATE_FRAME_FORMAT:
		 *((DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA *)pData) = pStreamState[StreamNumber].FrameFormat;
		break;
	case DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE:
		 *((DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA *)pData) = pStreamState[StreamNumber].InputColorSpace;
		break;
	case DXVAHD_STREAM_STATE_OUTPUT_RATE:
		 *((DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA *)pData) = pStreamState[StreamNumber].OutputRate;
		break;
	case DXVAHD_STREAM_STATE_SOURCE_RECT:
		 *((DXVAHD_STREAM_STATE_SOURCE_RECT_DATA *)pData) = pStreamState[StreamNumber].SourceRect;
		break;
	case DXVAHD_STREAM_STATE_DESTINATION_RECT:
		 *((DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA *)pData) = pStreamState[StreamNumber].DestinationRect;
		break;
	case DXVAHD_STREAM_STATE_ALPHA:
		 *((DXVAHD_STREAM_STATE_ALPHA_DATA *)pData) = pStreamState[StreamNumber].Alpha;
		break;
	case DXVAHD_STREAM_STATE_PALETTE:
		 *((DXVAHD_STREAM_STATE_PALETTE_DATA *)pData) = pStreamState[StreamNumber].Palette;
		break;
	case DXVAHD_STREAM_STATE_CLEAR_RECT:
		 *((DXVAHD_STREAM_STATE_CLEAR_RECT_DATA *)pData) = pStreamState[StreamNumber].ClearRect;
		break;
	case DXVAHD_STREAM_STATE_LUMA_KEY:
		 *((DXVAHD_STREAM_STATE_LUMA_KEY_DATA *)pData) = pStreamState[StreamNumber].LumaKey;
		break;
	case DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS:
		 *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData) = pStreamState[StreamNumber].Brightness;
		break;
	case DXVAHD_STREAM_STATE_FILTER_CONTRAST:
		 *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData) = pStreamState[StreamNumber].Contrast;
		break;
	case DXVAHD_STREAM_STATE_FILTER_HUE:
		 *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData) = pStreamState[StreamNumber].Hue;
		break;
	case DXVAHD_STREAM_STATE_FILTER_SATURATION:
		 *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData) = pStreamState[StreamNumber].Saturation;
		break;
	case DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION:
		 *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData) = pStreamState[StreamNumber].NoiseReduction;
		break;
	case DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT:
		 *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData) = pStreamState[StreamNumber].EdgeEnhancement;
		break;
	case DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING:
		 *((DXVAHD_STREAM_STATE_FILTER_DATA *)pData) = pStreamState[StreamNumber].AnamorphicScaling;
		break;
	}
	return S_OK;
}

HRESULT IDXVAHD_VideoProcessor::StealSurface(IDirect3DSurface9* lpDecode)
{
    //
    // draw the color bars
    //
    D3DLOCKED_RECT ddsd;
	int i;
    HRESULT hr = lpDecode->LockRect(&ddsd, NULL, D3DLOCK_NOSYSLOCK);
    if (hr == S_OK)
    {
        LPBYTE pDstOrg = (LPBYTE)ddsd.pBits;

        LPBYTE pDst = pDstOrg;
		memcpy(pDst, (void *)(&BltState), sizeof(DXVAHD_BLT_STATE_CONTENT));
		pDst += sizeof(DXVAHD_BLT_STATE_CONTENT);
		for (i = 0; i < DXVAHD_MAX_NUM_STREAMS; i++)
		{
			memcpy(pDst, (void *)(&(pStreamState[i])), sizeof(DXVAHD_STREAM_STATE_CONTENT));
			pDst += sizeof(DXVAHD_STREAM_STATE_CONTENT);
		}
        lpDecode->UnlockRect();
    }
	return hr;
}


HRESULT IDXVAHD_VideoProcessor::VideoProcessBltHD(
	IDirect3DSurface9* pOutputSurface,
	UINT               OutputFrame,
	UINT               StreamCount,
	DXVAHD_STREAM_DATA*  pData
)
{	// to be complete
	int i, j, k;
	void *pDst;

	StreamCount--;
// DXVA2.0
	for (i = 0; i < StreamCount; i++)
	{
		InputVideoSamples[i].Start = pData[i].InputVideoSamples.Start;
		InputVideoSamples[i].End = pData[i].InputVideoSamples.End;
		InputVideoSamples[i].SampleFormat = pData[i].InputVideoSamples.SampleFormat;
		InputVideoSamples[i].SrcSurface = pData[i].InputVideoSamples.SrcSurface;
		InputVideoSamples[i].SampleData = pData[i].InputVideoSamples.SampleData;
		InputVideoSamples[i].PlanarAlpha = pData[i].InputVideoSamples.PlanarAlpha;
		InputVideoSamples[i].SrcRect = pData[i].InputVideoSamples.SrcRect;
		InputVideoSamples[i].DstRect = pData[i].InputVideoSamples.DstRect;
	}

// DXVAHD
	// hack the second half of InputVideoSamples to hold StreamState and BltState
	// BltState
	pDst = (void *)(&(InputVideoSamples[StreamCount]));
	memcpy(pDst, (void *)(&BltState), sizeof(DXVAHD_BLT_STATE_CONTENT));
	for (i = 0; i < StreamCount; i++)
	{
		pDst = (void *)(&(InputVideoSamples[StreamCount + 1 + i]));
		memcpy(pDst, (void *)(&(pStreamState[i])), sizeof(DXVAHD_STREAM_STATE_CONTENT));
	}

	StealSurface(InputVideoSamples[0].SrcSurface);

	HRESULT hr = m_pVideodProcessDevice->VideoProcessBlt(
			pOutputSurface,
			&VideoProcessBltParams,
			&InputVideoSamples[0],
			StreamCount,
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

	pVideoProcessor->OnCreate();

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

