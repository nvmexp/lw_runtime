#ifndef __dxvahdapi_h__
#define __dxvahdapi_h__

#ifdef __cplusplus
extern "C"{
#endif 

typedef struct _DXVAHD_RATIONAL
{
    UINT Numerator;
    UINT Denominator;
} DXVAHD_RATIONAL;

typedef enum _DXVAHD_FRAME_FORMAT
{
    DXVAHD_FRAME_FORMAT_PROGRESSIVE                   = 0,
    DXVAHD_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST    = 1,
    DXVAHD_FRAME_FORMAT_INTERLACED_BOTTOM_FIELD_FIRST = 2
} DXVAHD_FRAME_FORMAT;

typedef enum _DXVAHD_DEVICE_FLAG
{
    DXVAHD_DEVICE_FLAG_PLAYBACK = 0x1,
    DXVAHD_DEVICE_FLAG_BATTERY  = 0x2,
    DXVAHD_DEVICE_FLAG_QUALITY  = 0x4
} DXVAHD_DEVICE_FLAG;

typedef enum _DXVAHD_SURFACE_TYPE
{
    DXVAHD_SURFACE_TYPE_VIDEO_INPUT         = 0,
    DXVAHD_SURFACE_TYPE_VIDEO_INPUT_PRIVATE = 1,
    DXVAHD_SURFACE_TYPE_VIDEO_OUTPUT        = 2
} DXVAHD_SURFACE_TYPE;

typedef enum _DXVAHD_DEVICE_TYPE
{
    DXVAHD_DEVICE_TYPE_HARDWARE        = 0,
    DXVAHD_DEVICE_TYPE_HARDWARE_DXVA2  = 1,
    DXVAHD_DEVICE_TYPE_HARDWARE_DXVA1  = 2,
    DXVAHD_DEVICE_TYPE_SOFTWARE        = 3,
    DXVAHD_DEVICE_TYPE_SOFTWARE_SHADER = 4,
    DXVAHD_DEVICE_TYPE_REFERENCE       = 5,
    DXVAHD_DEVICE_TYPE_OTHER           = 6
} DXVAHD_DEVICE_TYPE;

typedef enum _DXVAHD_DEVICE_CAPS
{
    DXVAHD_DEVICE_CAPS_LINEAR_SPACE = 0x1
} DXVAHD_DEVICE_CAPS;

typedef enum _DXVAHD_FEATURE_CAPS
{
    DXVAHD_FEATURE_CAPS_DOWNSAMPLE = 0x1,
    DXVAHD_FEATURE_CAPS_CLEAR_RECT = 0x2,
    DXVAHD_FEATURE_CAPS_LUMA_KEY   = 0x4
} DXVAHD_FEATURE_CAPS;

typedef enum _DXVAHD_FILTER_CAPS
{
    DXVAHD_FILTER_CAPS_PROCAMP          = 0x1,
    DXVAHD_FILTER_CAPS_NOISE_REDUCTION  = 0x2,
    DXVAHD_FILTER_CAPS_EDGE_ENHANCEMENT = 0x4
} DXVAHD_FILTER_CAPS;

typedef enum _DXVAHD_INPUT_FORMAT_CAPS
{
    DXVAHD_INPUT_FORMAT_CAPS_RGB_INTERLACED    = 0x1,
    DXVAHD_INPUT_FORMAT_CAPS_RGB_LIMITED_RANGE = 0x2,
    DXVAHD_INPUT_FORMAT_CAPS_RGB_PROCAMP       = 0x4,
    DXVAHD_INPUT_FORMAT_CAPS_RGB_LUMA_KEY      = 0x8
} DXVAHD_INPUT_FORMAT_CAPS;

typedef enum _DXVAHD_FILTER_RANGE
{
    DXVAHD_FILTER_RANGE_PROCAMP_BRIGHTNESS = 0,
    DXVAHD_FILTER_RANGE_PROCAMP_CONTRAST   = 1,
    DXVAHD_FILTER_RANGE_PROCAMP_HUE        = 2,
    DXVAHD_FILTER_RANGE_PROCAMP_SATURATION = 3,
    DXVAHD_FILTER_RANGE_NOISE_REDUCTION    = 4,
    DXVAHD_FILTER_RANGE_EDGE_ENHANCEMENT   = 5
} DXVAHD_FILTER_RANGE;

typedef enum _DXVAHD_PROCESSOR_CAPS
{
    DXVAHD_PROCESSOR_CAPS_DEINTERLACE_BLENDING            = 0x1,
    DXVAHD_PROCESSOR_CAPS_DEINTERLACE_BOB                 = 0x2,
    DXVAHD_PROCESSOR_CAPS_DEINTERLACE_ADAPTIVE            = 0x4,
    DXVAHD_PROCESSOR_CAPS_DEINTERLACE_MOTION_COMPENSATION = 0x8,
    DXVAHD_PROCESSOR_CAPS_ILWERSE_TELECINE                = 0x10,
    DXVAHD_PROCESSOR_CAPS_FRAME_RATE_COLWERSION           = 0x20
} DXVAHD_PROCESSOR_CAPS;

typedef enum _DXVAHD_ITELECINE_CAPS
{
    DXVAHD_ITELECINE_CAPS_32           = 0x1,
    DXVAHD_ITELECINE_CAPS_22           = 0x2,
    DXVAHD_ITELECINE_CAPS_2224         = 0x4,
    DXVAHD_ITELECINE_CAPS_2332         = 0x8,
    DXVAHD_ITELECINE_CAPS_32322        = 0x10,
    DXVAHD_ITELECINE_CAPS_55           = 0x20,
    DXVAHD_ITELECINE_CAPS_64           = 0x40,
    DXVAHD_ITELECINE_CAPS_87           = 0x80,
    DXVAHD_ITELECINE_CAPS_222222222223 = 0x100,
    DXVAHD_ITELECINE_CAPS_OTHER        = 0x80000000
} DXVAHD_ITELECINE_CAPS;

typedef enum _DXVAHD_ALPHA_FILL_MODE
{
    DXVAHD_ALPHA_FILL_MODE_BACKGROUND = 0,
    DXVAHD_ALPHA_FILL_MODE_ORIGINAL   = 1,
    DXVAHD_ALPHA_FILL_MODE_STREAM     = 2
} DXVAHD_ALPHA_FILL_MODE;

typedef enum _DXVAHD_OUTPUT_RATE
{
	DXVAHD_OUTPUT_RATE_NORMAL = 0,
	DXVAHD_OUTPUT_RATE_HALF = 1,
	DXVAHD_OUTPUT_RATE_LWSTOM = 2
} DXVAHD_OUTPUT_RATE;


typedef enum _DXVAHD_STREAM_STATE
{
    DXVAHD_STREAM_STATE_D3DFORMAT					= 0,
    DXVAHD_STREAM_STATE_FRAME_FORMAT				= 1,
    DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE			= 2,
    DXVAHD_STREAM_STATE_OUTPUT_RATE					= 3,
    DXVAHD_STREAM_STATE_SOURCE_RECT					= 4,
    DXVAHD_STREAM_STATE_DESTINATION_RECT			= 5,
    DXVAHD_STREAM_STATE_ALPHA						= 6,
    DXVAHD_STREAM_STATE_PALETTE						= 7,
    DXVAHD_STREAM_STATE_CLEAR_RECT					= 8,
    DXVAHD_STREAM_STATE_LUMA_KEY					= 9,
    DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS			= 10,
    DXVAHD_STREAM_STATE_FILTER_CONTRAST				= 11,
    DXVAHD_STREAM_STATE_FILTER_HUE					= 12,
    DXVAHD_STREAM_STATE_FILTER_SATURATION			= 13,
    DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION		= 14,
    DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT		= 15,
    DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING	= 16,
    DXVAHD_STREAM_STATE_PRIVATE						= 1000
} DXVAHD_STREAM_STATE;

typedef struct _DXVAHD_CONTENT_DESC
{
    D3DFORMAT           InputFormat;
    DXVAHD_FRAME_FORMAT InputFrameFormat;
    DXVAHD_RATIONAL     InputFrameRate;
    UINT                InputWidth;
    UINT                InputHeight;
    DXVAHD_RATIONAL     OutputFrameRate;
    UINT                OutputWidth;
    UINT                OutputHeight;
} DXVAHD_CONTENT_DESC;

typedef struct _DXVAHD_VPDEVCAPS
{
    DXVAHD_DEVICE_TYPE       DeviceType;
    DXVAHD_DEVICE_CAPS       DeviceCaps;
    DXVAHD_FEATURE_CAPS      FeatureCaps;
    DXVAHD_FILTER_CAPS       FilterCaps;
    DXVAHD_INPUT_FORMAT_CAPS InputFormatCaps;
    D3DPOOL                  InputPool;
} DXVAHD_VPDEVCAPS;

typedef struct _DXVAHD_FILTER_RANGE_DATA
{
    INT   Minimum;
    INT   Maximum;
    INT   Default;
    UINT  Step;
    FLOAT Multiplier;
} DXVAHD_FILTER_RANGE_DATA;

typedef struct _DXVAHD_VPCAPS
{
    GUID                  VPGuid;
    UINT                  BackwardReferenceFrames;
    UINT                  ForwardReferenceFrames;
    DXVAHD_PROCESSOR_CAPS ProcessorCaps;
    DXVAHD_ITELECINE_CAPS ITelecineCaps;
} DXVAHD_VPCAPS;

typedef struct _DXVAHD_FRAME_RATE
{
    DXVAHD_RATIONAL OutputRate;
    UINT            OutputFrames;
    BOOL            Interlaced;
    union
    {
        UINT        InputFrames;
        UINT        InputFields;
    };
} DXVAHD_FRAME_RATE;

typedef enum _DXVAHD_BLT_STATE
{
    DXVAHD_BLT_STATE_TARGET_RECT        = 0,
    DXVAHD_BLT_STATE_BACKGROUND_COLOR   = 1,
    DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE = 2,
    DXVAHD_BLT_STATE_ALPHA_FILL         = 3,
    DXVAHD_BLT_STATE_DOWNSAMPLE         = 4,
    DXVAHD_BLT_STATE_CLEAR_RECT         = 5,
    DXVAHD_BLT_STATE_PRIVATE            = 1000
} DXVAHD_BLT_STATE;

typedef struct _DXVAHD_BLT_STATE_TARGET_RECT_DATA
{
	BOOL Enable;
    RECT TargetRect;
} DXVAHD_BLT_STATE_TARGET_RECT_DATA;

typedef struct _DXVAHD_COLOR
{
    union
    {
        FLOAT R;
        FLOAT Y;
    };
    union
    {
        FLOAT G;
        FLOAT Cb;
    };
    union
    {
        FLOAT B;
        FLOAT Cr;
    };
    FLOAT A;
} DXVAHD_COLOR;

typedef struct _DXVAHD_BLT_STATE_BACKGROUND_COLOR_DATA
{
    BOOL         YCbCr;
    DXVAHD_COLOR BackgroundColor;
} DXVAHD_BLT_STATE_BACKGROUND_COLOR_DATA;

typedef struct _DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE_DATA
{
    UINT Usage        : 1;  // 0:Playback,     1:Processing
    UINT RGB_Range    : 1;  // 0:Full(0-255),  1:Limited(16-235)
    UINT YCbCr_Matrix : 1;  // 0:BT.601(SDTV), 1:BT.709(HDTV)
    UINT YCbCr_xvYCC  : 1;  // 0:Colwentional, 1:Expanded(xvYCC)
} DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE_DATA;

typedef struct _DXVAHD_BLT_STATE_ALPHA_FILL_DATA
{
    DXVAHD_ALPHA_FILL_MODE Mode;
    UINT                   StreamNumber;
} DXVAHD_BLT_STATE_ALPHA_FILL_DATA;

typedef struct _DXVAHD_BLT_STATE_DOWNSAMPLE_DATA
{
    BOOL Enable;
    SIZE Size;
} DXVAHD_BLT_STATE_DOWNSAMPLE_DATA;

typedef struct _DXVAHD_BLT_STATE_CLEAR_RECT_DATA
{
    BOOL Enable;
	RECT ClearRect[32];
} DXVAHD_BLT_STATE_CLEAR_RECT_DATA;

typedef struct _DXVAHD_BLT_STATE_PRIVATE_DATA
{
    GUID  Guid;
    UINT  DataSize;
    VOID* pData;
} DXVAHD_BLT_STATE_PRIVATE_DATA;

typedef struct _DXVAHD_STREAM_STATE_D3DFORMAT_DATA
{
    D3DFORMAT Format;
} DXVAHD_STREAM_STATE_D3DFORMAT_DATA;

typedef struct _DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA
{
    DXVAHD_FRAME_FORMAT FrameFormat;
} DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA;

typedef struct _DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA
{
    UINT RGB_Range    : 1;  // 0:Full(0-255),  1:Limited(16-235)
    UINT YCbCr_Matrix : 1;  // 0:BT.601(SDTV), 1:BT.709(HDTV)
    UINT YCbCr_xvYCC  : 1;  // 0:Colwentional, 1:Expanded(xvYCC)
} DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA;

typedef struct _DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA
{
    BOOL				RepeatFrame;
	DXVAHD_OUTPUT_RATE	OutputRate;
    DXVAHD_RATIONAL		LwstomRate;
} DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA;

typedef struct _DXVAHD_STREAM_STATE_SOURCE_RECT_DATA
{
	BOOL Enable;
    RECT SourceRect;
} DXVAHD_STREAM_STATE_SOURCE_RECT_DATA;

typedef struct _DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA
{
	BOOL Enable;
    RECT DestinationRect;
} DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA;

typedef struct _DXVAHD_STREAM_STATE_ALPHA_DATA
{
    BOOL  Enable;
    FLOAT Alpha;
} DXVAHD_STREAM_STATE_ALPHA_DATA;

typedef struct _DXVAHD_STREAM_STATE_PALETTE_DATA
{
	UINT Count;
    D3DCOLOR pEntries[256];
} DXVAHD_STREAM_STATE_PALETTE_DATA;

typedef struct _DXVAHD_STREAM_STATE_CLEAR_RECT_DATA
{
	DWORD ClearRectMask;
} DXVAHD_STREAM_STATE_CLEAR_RECT_DATA;

typedef struct _DXVAHD_STREAM_STATE_LUMA_KEY_DATA
{
    BOOL  Enable;
    FLOAT Lower;
    FLOAT Upper;
} DXVAHD_STREAM_STATE_LUMA_KEY_DATA;

typedef struct _DXVAHD_STREAM_STATE_FILTER_DATA
{
    BOOL Enable;
    INT  Level;
} DXVAHD_STREAM_STATE_FILTER_DATA;

typedef struct _DXVAHD_STREAM_STATE_PRIVATE_DATA
{
    GUID  Guid;
    UINT  DataSize;
    VOID* pData;
} DXVAHD_STREAM_STATE_PRIVATE_DATA;

typedef struct _DXVAHD_STREAM_DATA
{
    BOOL                Enable;
    union
    {
        UINT            InputFrame;
        UINT            InputField;
    };
    UINT                OutputIndex;
    UINT                BackwardReferenceFrames;
    UINT                ForwardReferenceFrames;
    IDirect3DSurface9** ppBackwardReferenceSurfaces;
    IDirect3DSurface9*  pInputSurface;
    IDirect3DSurface9** ppForwardReferenceSurfaces;
	// to borrow VideoProcessBlt
	DXVA2_VideoSample	InputVideoSamples;
} DXVAHD_STREAM_DATA;

typedef struct _DXVAHD_STREAM_STATE_CONTENT
{	// struct size: about 80 bytes
	DXVAHD_STREAM_STATE_D3DFORMAT_DATA D3DFormat;
	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA FrameFormat;
	DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA InputColorSpace;
    DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA OutputRate;
    DXVAHD_STREAM_STATE_SOURCE_RECT_DATA SourceRect;
    DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA DestinationRect;
	DXVAHD_STREAM_STATE_ALPHA_DATA Alpha;
	DXVAHD_STREAM_STATE_PALETTE_DATA Palette;
	DXVAHD_STREAM_STATE_CLEAR_RECT_DATA ClearRect;
	DXVAHD_STREAM_STATE_LUMA_KEY_DATA LumaKey;
	DXVAHD_STREAM_STATE_FILTER_DATA Brightness;
	DXVAHD_STREAM_STATE_FILTER_DATA Contrast;
	DXVAHD_STREAM_STATE_FILTER_DATA Hue;
	DXVAHD_STREAM_STATE_FILTER_DATA Saturation;
	DXVAHD_STREAM_STATE_FILTER_DATA NoiseReduction;
	DXVAHD_STREAM_STATE_FILTER_DATA EdgeEnhancement;
	DXVAHD_STREAM_STATE_FILTER_DATA AnamorphicScaling;
	DXVAHD_STREAM_STATE_PRIVATE_DATA Private;
} DXVAHD_STREAM_STATE_CONTENT;

typedef struct _DXVAHD_BLT_STATE_CONTENT
{	// struct size: about 32 bytes
	DXVAHD_BLT_STATE_TARGET_RECT_DATA TargetRect;
	DXVAHD_BLT_STATE_BACKGROUND_COLOR_DATA BackgroundColor;
	DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE_DATA OutputColorSpace;
	DXVAHD_BLT_STATE_ALPHA_FILL_DATA AlphaFill;
	DXVAHD_BLT_STATE_DOWNSAMPLE_DATA DownSample;
	DXVAHD_BLT_STATE_CLEAR_RECT_DATA ClearRect;
	DXVAHD_BLT_STATE_PRIVATE_DATA Private;
} DXVAHD_BLT_STATE_CONTENT;

#define DXVAHD_MAX_NUM_STREAMS	12
#if defined(__cplusplus) && !defined(CINTERFACE)
class IDXVAHD_VideoProcessor
{
public:
	// blt_state
	DXVAHD_BLT_STATE_CONTENT BltState;

	// stream_state
	DXVAHD_STREAM_STATE_CONTENT pStreamState[DXVAHD_MAX_NUM_STREAMS];

	// to borrow VideoProcessBlt
    IDirectXVideoProcessor*             m_pVideodProcessDevice;
	DXVA2_VideoProcessBltParams			VideoProcessBltParams;
	DXVA2_VideoSample InputVideoSamples[DXVAHD_MAX_NUM_STREAMS * 2 + 1];

	HRESULT OnCreate();

	HRESULT SetVideoProcessBltState(
		DXVAHD_BLT_STATE State,
		UINT             DataSize,
		VOID*            pData
	);

	HRESULT GetVideoProcessBltState(
		DXVAHD_BLT_STATE State,
		UINT             DataSize,
		VOID*            pData
	);

	HRESULT SetVideoProcessStreamState(
		UINT              StreamNumber,
		DXVAHD_STREAM_STATE State,
		UINT              DataSize,
		VOID*             pData
	);

	HRESULT GetVideoProcessStreamState(
		UINT              StreamNumber,
		DXVAHD_STREAM_STATE State,
		UINT              DataSize,
		VOID*             pData
	);

	HRESULT StealSurface(IDirect3DSurface9* lpDecode);

	HRESULT VideoProcessBltHD(
		IDirect3DSurface9* pOutputSurface,
		UINT               OutputFrame,
		UINT               StreamCount,
		DXVAHD_STREAM_DATA*  pData
	);
};

class IDXVAHD_Device
{
public:
	IDirectXVideoProcessorService* m_pAccelServices;

	HRESULT GetVideoProcessorOutputFormatCount(
		UINT* pCount
	);

	HRESULT GetVideoProcessorOutputFormats(
		UINT Count,
		D3DFORMAT* pFormat
	);

	HRESULT GetVideoProcessorInputFormatCount(
		UINT* pCount
	);

	HRESULT GetVideoProcessorInputFormats(
		UINT Count,
		D3DFORMAT* pFormat
	);

	HRESULT CreateVideoSurface(
		UINT                Width,
		UINT                Height,
		D3DFORMAT           Format,
		UINT                NumSurfaces,
		D3DPOOL             Pool,
		DXVAHD_SURFACE_TYPE Type,
		IDirect3DSurface9** ppSurface
	);

	HRESULT GetVideoProcessorDeviceCaps(
		DXVAHD_VPDEVCAPS* pCaps
	);

	HRESULT GetVideoProcessorFilterRange(
		DXVAHD_FILTER_RANGE       Filter,
		DXVAHD_FILTER_RANGE_DATA* pData
	);

	HRESULT GetVideoProcessorCapsCount(
		UINT* pCount
	);

	HRESULT GetVideoProcessorCaps(
		UINT Count,
		DXVAHD_VPCAPS* pCaps
	);

	HRESULT GetVideoProcessorFrameRateCount(
		GUID  VPGuid,
		UINT* pCount
	);

	HRESULT GetVideoProcessorFrameRates(
		GUID               VPGuid,
		UINT               Count,
		DXVAHD_FRAME_RATE* pFrameRate
	);

	HRESULT CreateVideoProcessor(
		GUID                  VPGuid,
		IDXVAHD_VideoProcessor** ppVideoProcessor
	);
};
	
#else 	/* C style interface */
#endif 	/* C style interface */

HRESULT DXVAHD_CreateDevice(
    IDirect3DDevice9Ex*  pD3DDevice,
    DXVAHD_CONTENT_DESC* pContentDesc,
    DXVAHD_DEVICE_FLAG   Flags,
    HMODULE              VideoProcessorPlugin,
	IDXVAHD_Device**     ppDevice
);


// added by DY
#define NEW_DXVA2
#define NEW_DXVA2_PADDING

//#define DXVAHD
// end of added by DY

struct CVideoData {
	BOOL								m_bDoCompositing;
    GUID                                m_guidVP;       // VideoProc device GUID.
    UINT                                m_startTime;    // Start time of the current video frame.
	// added by DY
#ifdef NEW_DXVA2
	int									m_hSrcMailwideo;
	int									m_hSrcSubVideo;
	int									m_hSrcSubTitle;
	int									m_hSrcGraphics;
	int									m_hSrcBackground;

	int									m_MailwideoFrameNo;
	int									m_SubVideoFrameNo;

	int									m_MailwideoMaxFrameNo;
	int									m_SubVideoMaxFrameNo;
#endif

    IDirect3D9Ex*						m_pD3DEx;
    IDirect3DDevice9Ex*					m_pD3DevEx;
	IDXVAHD_Device*						m_pHDDev;
	IDXVAHD_VideoProcessor*				m_pHDVP;
    //IDirectXVideoProcessorService*      m_pAccelServices;
	//IDirectXVideoProcessor*             m_pVideodProcessDevice;
    IDirect3DSurface9*                  m_pD3DRt;       // Back buffer surface.
    IDirect3DSurface9*                  m_pVideoMemMain;    // Surface where we draw the frame. 
	IDirect3DSurface9*                  m_pVideoMemSecondary;    // Surface where we draw the frame. 
	IDirect3DSurface9*                  m_pVideoMemInteractive;    // Surface where we draw the frame. 
	IDirect3DSurface9*                  m_pVideoMemGraphics;    // Surface where we draw the frame.
	IDirect3DSurface9*                  m_pVideoMemBackground;    // Surface where we draw the frame.
	IDirect3DSurface9*                  m_pVideoMemSubtitle;    // Surface where we draw the frame.
	IDirect3DSurface9*                  m_pSysMemSecondary; 

	// end of added by DY
};


#ifdef __cplusplus
}
#endif

#endif
