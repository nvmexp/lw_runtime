// dxvahdmfcView.cpp : implementation of the CdxvahdmfcView class
//

#include "stdafx.h"
#include <windows.h>    /* required for all Windows applications */
#include <windowsx.h>
#include <mmsystem.h>

#include <stdio.h>
#include <io.h>
#include <fcntl.h>

#include <initguid.h>
#include <d3d9.h>
#include <d3d9types.h>
#include <dxva2api.h>
#include "dxvahdapi.h"
#include "dxvahdmfc.h"

#include "dxvahdmfcDoc.h"
#include "dxvahdmfcView.h"
#include "ConfigDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CdxvahdmfcView

IMPLEMENT_DYNCREATE(CdxvahdmfcView, CView)

BEGIN_MESSAGE_MAP(CdxvahdmfcView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_WM_CREATE()
	ON_WM_TIMER()
	ON_COMMAND(ID_EDIT_CONFIG, &CdxvahdmfcView::OnEditConfig)
	ON_WM_KEYDOWN()
END_MESSAGE_MAP()

extern "C" void __stosd(PDWORD Destination, DWORD Value, SIZE_T Count);
#pragma intrinsic(__stosd)

template<typename T>
__inline void SAFE_RELEASE(T* &p)
{
    if (p)
    {
        p->Release();
        p = NULL;
    }
}

// -------------------------------------------------------------------------
// DXVA2 and Direct3D9 globals
// -------------------------------------------------------------------------
//
/* -------------------------------------------------------------------------
** Global variables
** -------------------------------------------------------------------------
*/
HINSTANCE   hInst;
HWND        hwndApp;
HANDLE      hEventNeverSignalled;  // Event handle used for waiting. 
BOOL        timerSet = FALSE;


DXVA2_ExtendedFormat    gSampleFormat;
DXVA2_ExtendedFormat    gSampleSSFormat;
DXVA2_ExtendedFormat    gDestFormat;

DXVA2_Fixed32           gBrightness = {0, 0};
DXVA2_Fixed32           gContrast = {0, 0};
DXVA2_Fixed32           gHue = {0, 0};
DXVA2_Fixed32           gSaturation = {0, 0};

DXVA2_Fixed32           gNoiseReduction = {0, 0};
DXVA2_Fixed32           gEdgeEnhance = {0, 0};
DXVA2_Fixed32           gNonLinearScale = {0, 0};

/* -------------------------------------------------------------------------
** Constants
** -------------------------------------------------------------------------
*/
const TCHAR szClassName[] = TEXT("VideoProcessor_CLASS");
const TCHAR szAppTitle[]  = TEXT("VideoProcessor Test Application");

const int cxVideo = 720;        // Width of the offscreen video surface.
const int cyVideo = 480;        // Height of the offscreen video surface.
const int cxHDVideo = 1920;
const int cyHDVideo = 1080;
const int insetPixels = 10;     // Size of the border in pixels.

const UINT fpsVideo = 4;                       // 25 Hz video
const UINT periodVideo = (1000 / fpsVideo);     // Frame duration in milliseconds.
const UINT periodVideo2 = (500 / fpsVideo);     // half the period

const D3DFORMAT yuvFormat = (D3DFORMAT)'2YUY';
const D3DFORMAT argbFormat = D3DFMT_A8R8G8B8;
const D3DFORMAT lw12Format = (D3DFORMAT)MAKEFOURCC('N','V','1','2');
const D3DFORMAT aip8Format = (D3DFORMAT)MAKEFOURCC('A','I','P','8');
const D3DFORMAT plffFormat = (D3DFORMAT)MAKEFOURCC('P','L','F','F');

#define COMPOSITION_LAYER_BACKGROUND		0x01
#define COMPOSITION_LAYER_MAILWIDEO			0x02
#define COMPOSITION_LAYER_SECONDARYVIDEO	0x04
#define COMPOSITION_LAYER_GRAPHICS			0x08
#define COMPOSITION_LAYER_SUBTITLE			0x10
#define COMPOSITION_LAYER_BITMAP_BACKGROUND	0x20

const DWORD dwPLFF[255] = {
0x00000000,     //   0   0   0
0x00800000,     // 128   0   0
0x00008000,     //   0 128   0
0x00000080,     //   0   0 128
0x00808000,     // 128 128   0
0x00800080,     // 128   0 128
0x00008080,     //   0 128 128
0x00808080,     // 128 128 128
0x00c0c0c0,     // 192 192 192
0x00ff0000,     // 255   0   0
0x0000ff00,     //   0 255   0
0x000000ff,     //   0   0 255
0x00ffff00,     // 255 255   0
0x00ff00ff,     // 255   0 255
0x0000ffff,     //   0 255 255
0x00ffffff,     // 255 255 255
};


const DWORD dwARGB[4] = {   //  colors in ARGB format
	D3DCOLOR_ARGB(255, 255, 0, 0),
	D3DCOLOR_ARGB(255, 0, 255, 0),
	D3DCOLOR_ARGB(255, 0, 0, 255),
	D3DCOLOR_ARGB(255, 50, 0, 100)
};

CVideoData gVideoData;

HRESULT CreateVideoProcessor(CVideoData *videoData);
int SetDXVAHDStates(CVideoData* videoData);

/*****************************Private*Routine******************************\
* CleanUpVideoResources
*
* Deletes any allocated resources.
*
\**************************************************************************/
void CleanUpVideoResources(CVideoData* videoData)
{
    SAFE_RELEASE(videoData->m_pD3DRt);
	SAFE_RELEASE(videoData->m_pVideoMemMain);
	SAFE_RELEASE(videoData->m_pVideoMemSecondary);
	SAFE_RELEASE(videoData->m_pVideoMemInteractive);
	SAFE_RELEASE(videoData->m_pVideoMemGraphics);
	SAFE_RELEASE(videoData->m_pVideoMemBackground);
	SAFE_RELEASE(videoData->m_pVideoMemSubtitle);
    SAFE_RELEASE(videoData->m_pHDVP->m_pVideodProcessDevice);
    SAFE_RELEASE(videoData->m_pHDDev->m_pAccelServices);
    SAFE_RELEASE(videoData->m_pD3DevEx);
    SAFE_RELEASE(videoData->m_pD3DEx);
	// added by DY
#ifdef NEW_DXVA2
	close(videoData->m_hSrcMailwideo);
	close(videoData->m_hSrcSubVideo);
#endif
	// end of added by DY
}

/*****************************Private*Routine******************************\
* CreateVideoResources()
*
* Creates the Video acceleration resources needed by the application.
*
\**************************************************************************/
BOOL CreateVideoResources(CVideoData* videoData, HWND hwnd)
{
    HRESULT hr = S_OK;
    IDirect3DSwapChain9* lpChain = NULL;
	D3DCAPS9 d3dcaps;
	DWORD BehaviorFlags = D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING;
	D3DPRESENT_PARAMETERS d3dpp;
	DXVAHD_CONTENT_DESC pContentDesc[7];
	GUID pVPGuid[16];

    D3DDISPLAYMODE displayMode;

    do
    {
        // Create an instance of Direct3D.
		hr = Direct3DCreate9Ex(D3D_SDK_VERSION, &videoData->m_pD3DEx);
        if (S_OK != hr) break;

        if (NULL == videoData->m_pD3DEx)
        {
            hr = E_FAIL;
            break;
        }

        // Find the current display mode of the default adapter.
        hr = videoData->m_pD3DEx->GetAdapterDisplayMode(D3DADAPTER_DEFAULT,
                                                      &displayMode);
        if (S_OK != hr)
        {
            break;
        }

        //
        // Enable HW Vertex Processing when it is available.
        //
        hr = videoData->m_pD3DEx->GetDeviceCaps(D3DADAPTER_DEFAULT,
                                              D3DDEVTYPE_HAL,
                                              &d3dcaps);

        if (S_OK != hr)
        {
            break;
        }

        if (d3dcaps.DevCaps & D3DDEVCAPS_HWTRANSFORMANDLIGHT)
        {
            BehaviorFlags |= D3DCREATE_HARDWARE_VERTEXPROCESSING;
        }
        else
        {
            BehaviorFlags |= D3DCREATE_SOFTWARE_VERTEXPROCESSING;
        }


        ZeroMemory(&d3dpp, sizeof(d3dpp));

        //
        // Create the Direct3D9 device and the swap chain. In this example, the swap 
        // chain is the same size as the current display mode. The format is RGB-32.
        //
        d3dpp.Windowed = TRUE;
        d3dpp.BackBufferFormat = D3DFMT_X8R8G8B8;
        d3dpp.BackBufferWidth = displayMode.Width;
        d3dpp.BackBufferHeight = displayMode.Height;
        d3dpp.BackBufferCount = 1;
        d3dpp.SwapEffect = D3DSWAPEFFECT_COPY;
        d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
        d3dpp.Flags = D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;

        hr = videoData->m_pD3DEx->CreateDeviceEx(D3DADAPTER_DEFAULT,
                                  D3DDEVTYPE_HAL,
                                  hwnd,
                                  BehaviorFlags,
                                  &d3dpp,
								  NULL,
                                  &videoData->m_pD3DevEx);

        if (hr != S_OK)
        {
            break;
        }

        // Create the DXVA service for video processing.
		hr = DXVAHD_CreateDevice(
				videoData->m_pD3DevEx,
				pContentDesc,
				DXVAHD_DEVICE_FLAG_PLAYBACK,
				NULL,
				&videoData->m_pHDDev);
		if (hr != S_OK) break;

// MAIN VIDEO
		hr = videoData->m_pHDDev->CreateVideoSurface(
					cxHDVideo,							// video frame width
					cyHDVideo,							// video frame height
					lw12Format,							// YUY2 format (4:2:2)
					1,
					D3DPOOL_DEFAULT,					// Default pool.
					DXVAHD_SURFACE_TYPE_VIDEO_INPUT,
					&videoData->m_pVideoMemMain);
			if (hr != S_OK) break;	

// SUBVIDEO
		hr = videoData->m_pHDDev->CreateVideoSurface(
				cxVideo,								// video frame width
				cyVideo,								// video frame height
				yuvFormat,								// YUY2 format (4:2:2)
				1,
				D3DPOOL_DEFAULT,						// Default pool.
				DXVAHD_SURFACE_TYPE_VIDEO_INPUT,
				&videoData->m_pVideoMemSecondary);
		if (hr != S_OK) break;

// GRAPHICS
		hr = videoData->m_pHDDev->CreateVideoSurface(
				cxHDVideo,								// video frame width
				cyHDVideo,								// video frame height
				argbFormat,								// ARGB format 
				1,
				D3DPOOL_DEFAULT,						// Default pool.
				DXVAHD_SURFACE_TYPE_VIDEO_INPUT,
				&videoData->m_pVideoMemGraphics);
		if (hr != S_OK) break;

// SUBTITLE
		hr = videoData->m_pHDDev->CreateVideoSurface(
				cxHDVideo,								// video frame width
				cyHDVideo,								// video frame height
				aip8Format,								// ARGB format 
				1,
				D3DPOOL_DEFAULT,						// Default pool.
				DXVAHD_SURFACE_TYPE_VIDEO_INPUT,
				&videoData->m_pVideoMemSubtitle);
        if (hr != S_OK) break;

// BITMAP BACKGROUND
		hr = videoData->m_pHDDev->CreateVideoSurface(
				cxHDVideo,								// video frame width
				cyHDVideo,								// video frame height
				argbFormat,								// ARGB format 
				1,
				D3DPOOL_DEFAULT,						// Default pool.
				DXVAHD_SURFACE_TYPE_VIDEO_INPUT,
				&videoData->m_pVideoMemBackground);
		if (hr != S_OK) break;

		// Get a pointer to the swap chain.
        hr = videoData->m_pD3DevEx->GetSwapChain(0, &lpChain);
		if (hr != S_OK) break;

        // Get a pointer to the first (and only) back buffer in the swap chain.
        hr = lpChain->GetBackBuffer(0, D3DBACKBUFFER_TYPE_MONO, &videoData->m_pD3DRt);
        if (hr != S_OK) break;
#if 0
		D3DSURFACE_DESC Desc;
		videoData->m_pD3DRt->GetDesc(&Desc);
#endif
        //
        // SDTV ITU-R BT.601 YCbCr to studio RGB [16...235]
        //
        gSampleFormat.SampleFormat = DXVA2_SampleProgressiveFrame;
        gSampleFormat.NominalRange = DXVA2_NominalRange_16_235;
        gSampleFormat.VideoTransferMatrix = DXVA2_VideoTransferMatrix_BT601;
		
        gSampleSSFormat.SampleFormat = DXVA2_SampleSubStream;
        gSampleSSFormat.NominalRange = DXVA2_NominalRange_16_235;
        gSampleSSFormat.VideoTransferMatrix = DXVA2_VideoTransferMatrix_BT601;

        gDestFormat.SampleFormat = DXVA2_SampleProgressiveFrame;
        gDestFormat.NominalRange = DXVA2_NominalRange_16_235;
        gDestFormat.VideoTransferMatrix = DXVA2_VideoTransferMatrix_BT709;

        // Create the video processor.
        hr = CreateVideoProcessor(videoData);
		// added by DY
#ifdef NEW_DXVA2
		// open source files
#ifdef NEW_DXVA2_PADDING
		videoData->m_hSrcMailwideo = _open("F:\\data\\video\\yuv\\Yozakurapad.lw12", _O_RDONLY | _O_BINARY);
		videoData->m_hSrcSubVideo = _open("F:\\data\\video\\yuv\\src19td_recpad.yuy2", _O_RDONLY | _O_BINARY);
#else
		videoData->m_hSrcMailwideo = _open("F:\\data\\video\\yuv\\Yozakura.lw12", _O_RDONLY | _O_BINARY);
		videoData->m_hSrcSubVideo = _open("F:\\data\\video\\yuv\\src19td_rec.yuy2", _O_RDONLY | _O_BINARY);
#endif
		videoData->m_MailwideoFrameNo = 0;
		videoData->m_SubVideoFrameNo = 0;
		videoData->m_MailwideoMaxFrameNo = 80;
		videoData->m_SubVideoMaxFrameNo = 80;
#endif
		// end of added by DY
    } while (0);

	SetDXVAHDStates(videoData);

    if (hr != S_OK)
    {
        CleanUpVideoResources(videoData);
    }

    SAFE_RELEASE(lpChain);


    return hr == S_OK;
}

/*****************************Private*Routine******************************\
* CreateVideoProcessor()
*
* Creates the Video Processor device.
*
\**************************************************************************/
HRESULT CreateVideoProcessor(CVideoData *videoData)
{

    //
    // Create the DXVA2 video processor
    //

    HRESULT hr = S_OK;

    UINT Count = 0;
    GUID* pGuids = NULL;    // Array of device GUIDs.

    DXVA2_VideoDesc videoDesc;    // Description of the video stream to process.

    videoDesc.SampleWidth  = cxHDVideo;
    videoDesc.SampleHeight = cyHDVideo,
    videoDesc.SampleFormat = gSampleFormat;
    videoDesc.Format = yuvFormat;
    videoDesc.InputSampleFreq.Numerator = fpsVideo;
    videoDesc.InputSampleFreq.Denominator = 1;
    videoDesc.OutputFrameFreq.Numerator = fpsVideo;
    videoDesc.OutputFrameFreq.Denominator = 1;

    DXVA2_ValueRange range = {0};

    do
    {
        // Get the supported device GUIDs.
        hr = videoData->m_pHDDev->m_pAccelServices->GetVideoProcessorDeviceGuids(
            &videoDesc,            // description of source video
            &Count,                // number of GUIDs
            &pGuids                // Receives the array of device GUIDs.
            );

        if (hr != S_OK)
        {
            break;
        }

        if (Count < 1)
        {
            hr = E_FAIL;
            break;
        }

        // Use the first GUID in the list.
        videoData->m_guidVP = pGuids[0];
#if 0
        for(int i = 0; i < (int)Count; i++){
            if(DXVA_BluraySuperbob == pGuids[i])
		    	videoData->m_guidVP = pGuids[i];
		}
#endif


        // 
        // Find the default value for brightness, contrast, hue,
        // and saturation. We will use these values when it is
        // time to blit the frame.

        hr = videoData->m_pHDDev->m_pAccelServices->GetProcAmpRange(
            videoData->m_guidVP,
            &videoDesc,
            D3DFMT_X8R8G8B8,
            DXVA2_ProcAmp_Brightness,
            &range);

        if (hr == S_OK)
        {
            gBrightness = range.DefaultValue;
        }

        hr = videoData->m_pHDDev->m_pAccelServices->GetProcAmpRange(
            videoData->m_guidVP,
            &videoDesc,
            D3DFMT_X8R8G8B8,
            DXVA2_ProcAmp_Contrast,
            &range);


        if (hr == S_OK)
        {
            gContrast = range.DefaultValue;
        }

        hr = videoData->m_pHDDev->m_pAccelServices->GetProcAmpRange(
            videoData->m_guidVP,
            &videoDesc,
            D3DFMT_X8R8G8B8,
            DXVA2_ProcAmp_Hue,
            &range);

        if (hr == S_OK)
        {
            gHue = range.DefaultValue;
        }

        hr = videoData->m_pHDDev->m_pAccelServices->GetProcAmpRange(
            videoData->m_guidVP,
            &videoDesc,
            D3DFMT_X8R8G8B8,
            DXVA2_ProcAmp_Saturation,
            &range);

        if (hr == S_OK)
        {
            gSaturation = range.DefaultValue;
        }

        // Create the video processor device.
		//pVPGuid[0] = DXVA_BluraySuperbob;
		hr = (videoData->m_pHDDev)->CreateVideoProcessor(
            (videoData->m_guidVP),
			&videoData->m_pHDVP);
    } while (0);

    CoTaskMemFree(pGuids);

    return hr;
}

/*****************************Private*Routine******************************\
* VideoProcessor_OnCreate
* 
* Called when the WM_CREATE message is received.
*
\**************************************************************************/
BOOL VideoProcessor_OnCreate(HWND hwnd, LPCREATESTRUCT lpCreateStruct)
{
    // Get the CVideoData pointer from the CREATESTRUCT and set it as 
    // window user data.

    CVideoData* videoData = (CVideoData*)lpCreateStruct->lpCreateParams;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)videoData);

    // Create the video processing resources.
    return CreateVideoResources(videoData, hwnd);
}

void PaintMailwideo(IDirect3DSurface9* lpDecode, UINT Tick, int w, int h, CVideoData* videoData)
{
    RECT rcSurf = {0, 0, w, h};
    D3DLOCKED_RECT ddsd;
	HRESULT hr = lpDecode->LockRect(&ddsd, NULL, D3DLOCK_NOSYSLOCK);
    if (hr == S_OK)
    {
        LPBYTE pDstOrg = (LPBYTE)ddsd.pBits;

        LPBYTE pDst = pDstOrg;
#ifdef NEW_DXVA2_PADDING
		read(videoData->m_hSrcMailwideo, pDst, ddsd.Pitch * h);
		pDst += ddsd.Pitch * h;
#else
        for (int y = 0; y < h; y++)
        {
			//DWORD dwFill;
			//dwFill = (y * 255 / h);
			//dwFill = (dwFill << 24) | (dwFill << 16) | (dwFill << 8) | dwFill;;
			//__stosd((PDWORD)pDst, dwFill, w>>2);
			// added by DY
#ifdef NEW_DXVA2
			read(videoData->m_hSrcMailwideo, pDst, w);
#endif
			// end of added by DY
            pDst += ddsd.Pitch;
        }
#endif
#ifdef NEW_DXVA2_PADDING
		read(videoData->m_hSrcMailwideo, pDst, ddsd.Pitch * (h >> 1));
		pDst += ddsd.Pitch * h;
#else
		for (int y = 0; y < (h>>1); y++)
        {
            //DWORD dwFill = dwLW12UV[(y * 16) / (h >>1)];
            DWORD dwFill = 0x80808080;
            //__stosd((PDWORD)pDst, dwFill, w>>2);

			// added by DY
#ifdef NEW_DXVA2
			//lseek(videoData->m_hSrcMailwideo, w, SEEK_LWR);
			read(videoData->m_hSrcMailwideo, pDst, w);
#endif
			// end of added by DY
            pDst += ddsd.Pitch;
        }
#endif
		// added by DY
#ifdef NEW_DXVA2
		videoData->m_MailwideoFrameNo++;
		if (videoData->m_MailwideoFrameNo >= videoData->m_MailwideoMaxFrameNo)
		{
			lseek(videoData->m_hSrcMailwideo, 0L, SEEK_SET);
			videoData->m_MailwideoFrameNo = 0;
		}
#endif
		// end of added by DY
        lpDecode->UnlockRect();
    }
}

/*****************************Private*Routine******************************\
* PaintSubVideo
*
* Draws color bars in YUY2 format on the Direct3D surface.
*
\**************************************************************************/
void PaintSubVideo(IDirect3DSurface9* lpDecode, UINT Tick, int w, int h,	CVideoData* videoData)
{
    RECT rcSurf = {0, 0, w, h};
    D3DLOCKED_RECT ddsd;
    HRESULT hr = lpDecode->LockRect(&ddsd, NULL, D3DLOCK_NOSYSLOCK);
    if (hr == S_OK)
    {
        LPBYTE pDstOrg = (LPBYTE)ddsd.pBits;

        LPBYTE pDst = pDstOrg;
#ifdef NEW_DXVA2_PADDING
		read(videoData->m_hSrcSubVideo, pDst, ddsd.Pitch * h);
#else
        for (int y = 0; y < h; y++)
        {
			//DWORD dwFill = 0x804f804f;		// grey
            //__stosd((PDWORD)pDst, dwFill, w>>1);
#ifdef NEW_DXVA2
			read(videoData->m_hSrcSubVideo, pDst, w << 1);
#endif
            pDst += ddsd.Pitch;
        }
#endif
		// added by DY
#ifdef NEW_DXVA2
		videoData->m_SubVideoFrameNo++;
		if (videoData->m_SubVideoFrameNo >= videoData->m_SubVideoMaxFrameNo)
		{
			lseek(videoData->m_hSrcSubVideo, 0L, SEEK_SET);
			videoData->m_SubVideoFrameNo = 0;
		}
#endif
		// end of added by DY
        lpDecode->UnlockRect();
    }
}

void PaintGraphics(IDirect3DSurface9* lpDecode, UINT Tick, int w, int h)
{
    //
    // draw the color bars
    //
	DWORD pData[2048];
    D3DLOCKED_RECT ddsd;
    HRESULT hr = lpDecode->LockRect(&ddsd, NULL, D3DLOCK_NOSYSLOCK);
    if (hr == S_OK)
    {
        LPBYTE pDstOrg = (LPBYTE)ddsd.pBits;

        LPBYTE pDst = pDstOrg;
        for (int y = 0; y < h; y++)
        {
			DWORD dwFill;
			int x;
#if 0
			dwFill = 0x000000f0;	// 0xARGB
            __stosd((PDWORD)pDst, dwFill, w);
#else
			for (x = 0; x < 256; x++)
			{
				pData[x] = x;
			}
			for (x = 0; x < 256; x++)
			{
				pData[x + 256] = x << 8;
			}
			for (x = 0; x < 256; x++)
			{
				pData[x + 512] = x << 16;
			}
			for (x = 0; x < 256; x++)
			{
				pData[x + 768] = (x << 16) + (x << 8) + x;
			}
			for (x = 0; x < 256; x++)
			{
				pData[x + 1024] = ((255- x) << 16) + ((255- x) << 8) + (255- x);
			}
			for (x = 0; x < 256; x++)
			{
				pData[x + 1280] = (x << 16) + (x << 8);
			}
			for (x = 0; x < 256; x++)
			{
				pData[x + 1536] = (x << 16) + x;
			}
			for (x = 0; x < 256; x++)
			{
				pData[x + 1792] = (x << 8) + x;
			}
			memcpy(pDst, pData, w * 4);
#endif
            pDst += ddsd.Pitch;
        }

        lpDecode->UnlockRect();
    }
}

void PaintSubtitle(IDirect3DSurface9* lpDecode, UINT Tick, int w, int h)
{
    // AIP8 format
    // draw the color bars
    //
    D3DLOCKED_RECT ddsd;
    HRESULT hr = lpDecode->LockRect(&ddsd, NULL, D3DLOCK_NOSYSLOCK);
    if (hr == S_OK)
    {
        LPBYTE pDstOrg = (LPBYTE)ddsd.pBits;

        LPBYTE pDst = pDstOrg;
        for (int y = 0; y < h; y++)
        {
			BYTE bFill = y % 256;
			DWORD dwFill = (bFill << 24) + (bFill << 16) + (bFill << 8) + bFill;
			//DWORD dwFill = 0x02030405;
            __stosd((PDWORD)pDst, dwFill, w >> 2);
            pDst += ddsd.Pitch;
        }
	
        lpDecode->UnlockRect();
    }
}

void PaintBitmapBackground(IDirect3DSurface9* lpDecode, UINT Tick, int w, int h)
{
    RECT rcSurf = {0, 0, w, h};
    RECT ball = {0, 0, 32, 32};
    int Offset = (Tick * 2) % w;
    OffsetRect(&ball, Offset, 0);
    IntersectRect(&ball, &ball, &rcSurf);

    //
    // draw the color bars
    //
    D3DLOCKED_RECT ddsd;
    HRESULT hr = lpDecode->LockRect(&ddsd, NULL, D3DLOCK_NOSYSLOCK);
    if (hr == S_OK)
    {
        LPBYTE pDstOrg = (LPBYTE)ddsd.pBits;

        LPBYTE pDst = pDstOrg;
        for (int y = 0; y < h; y++)
        {
			DWORD dwFill = dwARGB[3];
            //__stosd((PDWORD)pDst, dwFill, w);
            pDst += ddsd.Pitch;
        }

        lpDecode->UnlockRect();
    }
}

int SetDXVAHDStates(CVideoData* videoData)
{
    // Set up the DXVA2_VideoSample structure to hold information
    // about the video frame.
	int i;
	DWORD dwNumSurfs = 0;

#if 1
// MAIN VIDEO
	DXVAHD_STREAM_STATE_D3DFORMAT_DATA d3dFormatMailwideo;
	d3dFormatMailwideo.Format = lw12Format;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_D3DFORMAT,
		sizeof(DXVAHD_STREAM_STATE_D3DFORMAT_DATA),
		(void *)(&d3dFormatMailwideo));

	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA frameFormatMailwideo;
#if 1
	frameFormatMailwideo.FrameFormat = DXVAHD_FRAME_FORMAT_PROGRESSIVE;
#else
	frameFormatMailwideo.FrameFormat = DXVAHD_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FRAME_FORMAT,
		sizeof(DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA),
		(void *)(&frameFormatMailwideo));

	DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA inputColorSpaceDataMailwideo;
    inputColorSpaceDataMailwideo.RGB_Range = 1;
    inputColorSpaceDataMailwideo.YCbCr_Matrix = 0;  // 0:BT.601(SDTV), 1:BT.709(HDTV)
    inputColorSpaceDataMailwideo.YCbCr_xvYCC = 0;  // 0:Colwentional, 1:Expanded(xvYCC)
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE,
		sizeof(DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA),
		(void *)(&inputColorSpaceDataMailwideo));

	DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA outputRateMailwideo;
	outputRateMailwideo.RepeatFrame = FALSE;
	outputRateMailwideo.OutputRate = DXVAHD_OUTPUT_RATE_NORMAL;
	outputRateMailwideo.LwstomRate.Denominator = 1;
	outputRateMailwideo.LwstomRate.Numerator = 1;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_OUTPUT_RATE,
		sizeof(DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA),
		(void *)(&outputRateMailwideo));

	DXVAHD_STREAM_STATE_SOURCE_RECT_DATA srcRectMailwideo;
	srcRectMailwideo.Enable = TRUE;
	SetRect(&(srcRectMailwideo.SourceRect), 0, 0, cxHDVideo, cyHDVideo);
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_SOURCE_RECT,
		sizeof(DXVAHD_STREAM_STATE_SOURCE_RECT_DATA),
		(void *)(&srcRectMailwideo));

	DXVAHD_STREAM_STATE_ALPHA_DATA alphaMailwideo;
#if 1
	alphaMailwideo.Enable = FALSE;
	alphaMailwideo.Alpha = 1.;
#else
	alphaMailwideo.Enable = TRUE;
	alphaMailwideo.Alpha = 1.;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaMailwideo));

	DXVAHD_STREAM_STATE_PALETTE_DATA paletteMailwideo;
	paletteMailwideo.Count = 0;
#if 0
	paletteMailwideo.pEntries = (D3DCOLOR *)calloc(paletteMailwideo.Count, sizeof(D3DCOLOR));
	for (i = 0; i < paletteMailwideo.Count; i++)
	{
		paletteMailwideo.pEntries[i] = D3DCOLOR_ARGB(255, 255, 0, 0);
	}
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_PALETTE,
		sizeof(DXVAHD_STREAM_STATE_PALETTE_DATA),
		(void *)(&paletteMailwideo));

	DXVAHD_STREAM_STATE_CLEAR_RECT_DATA clearRectMailwideo;
	clearRectMailwideo.ClearRectMask = 0;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_CLEAR_RECT,
		sizeof(DXVAHD_STREAM_STATE_CLEAR_RECT_DATA),
		(void *)(&clearRectMailwideo));

	DXVAHD_STREAM_STATE_LUMA_KEY_DATA lumaKeyMailwideo;
#if 1
	lumaKeyMailwideo.Enable = FALSE;
	lumaKeyMailwideo.Lower = 0.;
	lumaKeyMailwideo.Upper = 0.;
#else
	lumaKeyMailwideo.Enable = TRUE;
	lumaKeyMailwideo.Lower = 0.;
	lumaKeyMailwideo.Upper = 0.;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_LUMA_KEY,
		sizeof(DXVAHD_STREAM_STATE_LUMA_KEY_DATA),
		(void *)(&lumaKeyMailwideo));

	DXVAHD_STREAM_STATE_FILTER_DATA brightnessMailwideo;
#if 0
	brightnessMailwideo.Enable = FALSE;
	brightnessMailwideo.Level = 128;
#else
	brightnessMailwideo.Enable = TRUE;
	brightnessMailwideo.Level = 128;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessMailwideo));

	DXVAHD_STREAM_STATE_FILTER_DATA contrastMailwideo;
#if 0
	contrastMailwideo.Enable = FALSE;
	contrastMailwideo.Level = 128;
#else
	contrastMailwideo.Enable = TRUE;
	contrastMailwideo.Level = 128;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastMailwideo));

	DXVAHD_STREAM_STATE_FILTER_DATA hueMailwideo;
#if 0
	hueMailwideo.Enable = FALSE;
	hueMailwideo.Level = 128;
#else
	hueMailwideo.Enable = TRUE;
	hueMailwideo.Level = 128;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueMailwideo));

	DXVAHD_STREAM_STATE_FILTER_DATA saturationMailwideo;
#if 0
	saturationMailwideo.Enable = FALSE;
	saturationMailwideo.Level = 128;
#else
	saturationMailwideo.Enable = TRUE;
	saturationMailwideo.Level = 128;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationMailwideo));

	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionMailwideo;
#if 1
	noiseReductionMailwideo.Enable = FALSE;
	noiseReductionMailwideo.Level = 0;
#else
	noiseReductionMailwideo.Enable = TRUE;
	noiseReductionMailwideo.Level = 0;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionMailwideo));

	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementMailwideo;
#if 0
	edgeEnhancementMailwideo.Enable = FALSE;
	edgeEnhancementMailwideo.Level = 0;
#else
	edgeEnhancementMailwideo.Enable = TRUE;
	edgeEnhancementMailwideo.Level = 0;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementMailwideo));

	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingMailwideo;
#if 1
	anamorphicScalingMailwideo.Enable = FALSE;
	anamorphicScalingMailwideo.Level = 32;
#else
	anamorphicScalingMailwideo.Enable = TRUE;
	anamorphicScalingMailwideo.Level = 64;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingMailwideo));

	dwNumSurfs++;
#endif

#if 1
// SECONDARY VIDEO
	DXVAHD_STREAM_STATE_D3DFORMAT_DATA d3dFormatSubVideo;
	d3dFormatSubVideo.Format = yuvFormat;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_D3DFORMAT,
		sizeof(DXVAHD_STREAM_STATE_D3DFORMAT_DATA),
		(void *)(&d3dFormatSubVideo));

	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA frameFormatSubVideo;
#if 0
	frameFormatSubVideo.FrameFormat = DXVAHD_FRAME_FORMAT_PROGRESSIVE;
#else
	frameFormatSubVideo.FrameFormat = DXVAHD_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FRAME_FORMAT,
		sizeof(DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA),
		(void *)(&frameFormatSubVideo));

	DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA inputColorSpaceDataSubVideo;
    inputColorSpaceDataSubVideo.RGB_Range = 1;
    inputColorSpaceDataSubVideo.YCbCr_Matrix = 0;  // 0:BT.601(SDTV), 1:BT.709(HDTV)
    inputColorSpaceDataSubVideo.YCbCr_xvYCC = 0;  // 0:Colwentional, 1:Expanded(xvYCC)
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE,
		sizeof(DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA),
		(void *)(&inputColorSpaceDataSubVideo));

	DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA outputRateSubVideo;
	outputRateSubVideo.RepeatFrame = FALSE;
	outputRateSubVideo.OutputRate = DXVAHD_OUTPUT_RATE_NORMAL;
	outputRateSubVideo.LwstomRate.Denominator = 1;
	outputRateSubVideo.LwstomRate.Numerator = 1;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_OUTPUT_RATE,
		sizeof(DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA),
		(void *)(&outputRateSubVideo));

	DXVAHD_STREAM_STATE_SOURCE_RECT_DATA srcRectSubVideo;
	srcRectSubVideo.Enable = TRUE;
	SetRect(&(srcRectSubVideo.SourceRect), 0, 0, cxVideo, cyVideo);
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_SOURCE_RECT,
		sizeof(DXVAHD_STREAM_STATE_SOURCE_RECT_DATA),
		(void *)(&srcRectSubVideo));

	DXVAHD_STREAM_STATE_ALPHA_DATA alphaSubVideo;
#if 1
	alphaSubVideo.Enable = FALSE;
	alphaSubVideo.Alpha = 1.;
#else
	alphaSubVideo.Enable = TRUE;
	alphaSubVideo.Alpha = .5;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaSubVideo));

	DXVAHD_STREAM_STATE_PALETTE_DATA paletteSubVideo;
	paletteSubVideo.Count = 0;
#if 0
	paletteSubVideo.pEntries = (D3DCOLOR *)calloc(paletteSubVideo.Count, sizeof(D3DCOLOR));
	for (i = 0; i < paletteSubVideo.Count; i++)
	{
		paletteSubVideo.pEntries[i] = D3DCOLOR_ARGB(255, 255, 0, 0);
	}
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_PALETTE,
		sizeof(DXVAHD_STREAM_STATE_PALETTE_DATA),
		(void *)(&paletteSubVideo));

	DXVAHD_STREAM_STATE_CLEAR_RECT_DATA clearRectSubVideo;
	clearRectSubVideo.ClearRectMask = 1;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_CLEAR_RECT,
		sizeof(DXVAHD_STREAM_STATE_CLEAR_RECT_DATA),
		(void *)(&clearRectSubVideo));

	DXVAHD_STREAM_STATE_LUMA_KEY_DATA lumaKeySubVideo;
	lumaKeySubVideo.Enable = FALSE;
	lumaKeySubVideo.Lower = 0.;
	lumaKeySubVideo.Upper = 0.;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_LUMA_KEY,
		sizeof(DXVAHD_STREAM_STATE_LUMA_KEY_DATA),
		(void *)(&lumaKeySubVideo));

	DXVAHD_STREAM_STATE_FILTER_DATA brightnessSubVideo;
#if 0
	brightnessSubVideo.Enable = FALSE;
	brightnessSubVideo.Level = 128;
#else
	brightnessSubVideo.Enable = TRUE;
	brightnessSubVideo.Level = 255;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessSubVideo));

	DXVAHD_STREAM_STATE_FILTER_DATA contrastSubVideo;
#if 1
	contrastSubVideo.Enable = FALSE;
	contrastSubVideo.Level = 128;
#else
	contrastSubVideo.Enable = TRUE;
	contrastSubVideo.Level = 128;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastSubVideo));

	DXVAHD_STREAM_STATE_FILTER_DATA hueSubVideo;
#if 1
	hueSubVideo.Enable = FALSE;
	hueSubVideo.Level = 128;
#else
	hueSubVideo.Enable = TRUE;
	hueSubVideo.Level = 128;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueSubVideo));

	DXVAHD_STREAM_STATE_FILTER_DATA saturationSubVideo;
	saturationSubVideo.Enable = FALSE;
	saturationSubVideo.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationSubVideo));

	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionSubVideo;
#if 1
	noiseReductionSubVideo.Enable = FALSE;
	noiseReductionSubVideo.Level = 128;
#else
	noiseReductionSubVideo.Enable = TRUE;
	noiseReductionSubVideo.Level = 128;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionSubVideo));

	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementSubVideo;
#if 1
	edgeEnhancementSubVideo.Enable = FALSE;
	edgeEnhancementSubVideo.Level = 128;
#else
	edgeEnhancementSubVideo.Enable = TRUE;
	edgeEnhancementSubVideo.Level = 128;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementSubVideo));

	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingSubVideo;
#if 1
	anamorphicScalingSubVideo.Enable = FALSE;
	anamorphicScalingSubVideo.Level = 32;
#else
	anamorphicScalingSubVideo.Enable = TRUE;
	anamorphicScalingSubVideo.Level = 64;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingSubVideo));

	dwNumSurfs++;
#endif

#if 1
// GRAPHICS
	DXVAHD_STREAM_STATE_D3DFORMAT_DATA d3dFormatGraphics;
	d3dFormatGraphics.Format = argbFormat;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_D3DFORMAT,
		sizeof(DXVAHD_STREAM_STATE_D3DFORMAT_DATA),
		(void *)(&d3dFormatGraphics));

	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA frameFormatGraphics;
	frameFormatGraphics.FrameFormat = DXVAHD_FRAME_FORMAT_PROGRESSIVE;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FRAME_FORMAT,
		sizeof(DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA),
		(void *)(&frameFormatGraphics));


	DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA inputColorSpaceDataGraphics;
    inputColorSpaceDataGraphics.RGB_Range = 1;
    inputColorSpaceDataGraphics.YCbCr_Matrix = 0;  // 0:BT.601(SDTV), 1:BT.709(HDTV)
    inputColorSpaceDataGraphics.YCbCr_xvYCC = 0;  // 0:Colwentional, 1:Expanded(xvYCC)
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE,
		sizeof(DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA),
		(void *)(&inputColorSpaceDataGraphics));

	DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA outputRateGraphics;
	outputRateGraphics.RepeatFrame = FALSE;
	outputRateGraphics.OutputRate = DXVAHD_OUTPUT_RATE_NORMAL;
	outputRateGraphics.LwstomRate.Denominator = 1;
	outputRateGraphics.LwstomRate.Numerator = 1;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_OUTPUT_RATE,
		sizeof(DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA),
		(void *)(&outputRateGraphics));

	DXVAHD_STREAM_STATE_SOURCE_RECT_DATA srcRectGraphics;
	srcRectGraphics.Enable = TRUE;
	SetRect(&(srcRectGraphics.SourceRect), 0, cyHDVideo * 4/5, cxHDVideo, cyHDVideo * 9/10);
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_SOURCE_RECT,
		sizeof(DXVAHD_STREAM_STATE_SOURCE_RECT_DATA),
		(void *)(&srcRectGraphics));

	DXVAHD_STREAM_STATE_ALPHA_DATA alphaGraphics;
#if 0
	alphaGraphics.Enable = FALSE;
	alphaGraphics.Alpha = 1.;
#else
	alphaGraphics.Enable = TRUE;
	alphaGraphics.Alpha = 0.5;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaGraphics));

	DXVAHD_STREAM_STATE_PALETTE_DATA paletteGraphics;
	paletteGraphics.Count = 0;
#if 0
	paletteGraphics.pEntries = (D3DCOLOR *)calloc(paletteGraphics.Count, sizeof(D3DCOLOR));
	for (i = 0; i < paletteGraphics.Count; i++)
	{
		paletteGraphics.pEntries[i] = D3DCOLOR_ARGB(255, 255, 0, 0);
	}
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_PALETTE,
		sizeof(DXVAHD_STREAM_STATE_PALETTE_DATA),
		(void *)(&paletteGraphics));

	DXVAHD_STREAM_STATE_CLEAR_RECT_DATA clearRectGraphics;
	clearRectGraphics.ClearRectMask = 0;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_CLEAR_RECT,
		sizeof(DXVAHD_STREAM_STATE_CLEAR_RECT_DATA),
		(void *)(&clearRectGraphics));

	DXVAHD_STREAM_STATE_LUMA_KEY_DATA lumaKeyGraphics;
	lumaKeyGraphics.Enable = FALSE;
	lumaKeyGraphics.Lower = 0.;
	lumaKeyGraphics.Upper = 0.;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_LUMA_KEY,
		sizeof(DXVAHD_STREAM_STATE_LUMA_KEY_DATA),
		(void *)(&lumaKeyGraphics));

	DXVAHD_STREAM_STATE_FILTER_DATA brightnessGraphics;
	brightnessGraphics.Enable = FALSE;
	brightnessGraphics.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessGraphics));

	DXVAHD_STREAM_STATE_FILTER_DATA contrastGraphics;
	contrastGraphics.Enable = FALSE;
	contrastGraphics.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastGraphics));

	DXVAHD_STREAM_STATE_FILTER_DATA hueGraphics;
	hueGraphics.Enable = FALSE;
	hueGraphics.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueGraphics));

	DXVAHD_STREAM_STATE_FILTER_DATA saturationGraphics;
	saturationGraphics.Enable = FALSE;
	saturationGraphics.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationGraphics));

	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionGraphics;
	noiseReductionGraphics.Enable = FALSE;
	noiseReductionGraphics.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionGraphics));

	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementGraphics;
	edgeEnhancementGraphics.Enable = FALSE;
	edgeEnhancementGraphics.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementGraphics));

	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingGraphics;
	anamorphicScalingGraphics.Enable = FALSE;
	anamorphicScalingGraphics.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingGraphics));

	dwNumSurfs++;
#endif

#if 1
// SUBTITLE
	DXVAHD_STREAM_STATE_D3DFORMAT_DATA d3dFormatSubtitle;
	d3dFormatSubtitle.Format = aip8Format;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_D3DFORMAT,
		sizeof(DXVAHD_STREAM_STATE_D3DFORMAT_DATA),
		(void *)(&d3dFormatSubtitle));

	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA frameFormatSubtitle;
	frameFormatSubtitle.FrameFormat = DXVAHD_FRAME_FORMAT_PROGRESSIVE;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FRAME_FORMAT,
		sizeof(DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA),
		(void *)(&frameFormatSubtitle));


	DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA inputColorSpaceDataSubtitle;
    inputColorSpaceDataSubtitle.RGB_Range = 1;
    inputColorSpaceDataSubtitle.YCbCr_Matrix = 0;  // 0:BT.601(SDTV), 1:BT.709(HDTV)
    inputColorSpaceDataSubtitle.YCbCr_xvYCC = 0;  // 0:Colwentional, 1:Expanded(xvYCC)
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE,
		sizeof(DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA),
		(void *)(&inputColorSpaceDataSubtitle));

	DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA outputRateSubtitle;
	outputRateSubtitle.RepeatFrame = FALSE;
	outputRateSubtitle.OutputRate = DXVAHD_OUTPUT_RATE_NORMAL;
	outputRateSubtitle.LwstomRate.Denominator = 1;
	outputRateSubtitle.LwstomRate.Numerator = 1;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_OUTPUT_RATE,
		sizeof(DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA),
		(void *)(&outputRateSubtitle));

	DXVAHD_STREAM_STATE_SOURCE_RECT_DATA srcRectSubtitle;
	srcRectSubtitle.Enable = TRUE;
	SetRect(&(srcRectSubtitle.SourceRect), 0, 0, cxHDVideo, cyHDVideo);
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_SOURCE_RECT,
		sizeof(DXVAHD_STREAM_STATE_SOURCE_RECT_DATA),
		(void *)(&srcRectSubtitle));

	DXVAHD_STREAM_STATE_ALPHA_DATA alphaSubtitle;
#if 0
	alphaSubtitle.Enable = FALSE;
	alphaSubtitle.Alpha = 1.;
#else
	alphaSubtitle.Enable = TRUE;
	alphaSubtitle.Alpha = 0.5;
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaSubtitle));

	DXVAHD_STREAM_STATE_PALETTE_DATA paletteSubtitle;
	paletteSubtitle.Count = 256;
#if 1
	//paletteSubtitle.pEntries = (D3DCOLOR *)calloc(paletteSubtitle.Count, sizeof(D3DCOLOR));
	for (i = 0; i < paletteSubtitle.Count; i++)
	{
		paletteSubtitle.pEntries[i] = dwPLFF[i % 16];		//D3DCOLOR_ARGB(255, 255, 0, 0);
	}
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_PALETTE,
		sizeof(DXVAHD_STREAM_STATE_PALETTE_DATA),
		(void *)(&paletteSubtitle));

	DXVAHD_STREAM_STATE_CLEAR_RECT_DATA clearRectSubtitle;
	clearRectSubtitle.ClearRectMask = 0;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_CLEAR_RECT,
		sizeof(DXVAHD_STREAM_STATE_CLEAR_RECT_DATA),
		(void *)(&clearRectSubtitle));

	DXVAHD_STREAM_STATE_LUMA_KEY_DATA lumaKeySubtitle;
	lumaKeySubtitle.Enable = FALSE;
	lumaKeySubtitle.Lower = 0.;
	lumaKeySubtitle.Upper = 0.;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_LUMA_KEY,
		sizeof(DXVAHD_STREAM_STATE_LUMA_KEY_DATA),
		(void *)(&lumaKeySubtitle));

	DXVAHD_STREAM_STATE_FILTER_DATA brightnessSubtitle;
	brightnessSubtitle.Enable = FALSE;
	brightnessSubtitle.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessSubtitle));

	DXVAHD_STREAM_STATE_FILTER_DATA contrastSubtitle;
	contrastSubtitle.Enable = FALSE;
	contrastSubtitle.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastSubtitle));

	DXVAHD_STREAM_STATE_FILTER_DATA hueSubtitle;
	hueSubtitle.Enable = FALSE;
	hueSubtitle.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueSubtitle));

	DXVAHD_STREAM_STATE_FILTER_DATA saturationSubtitle;
	saturationSubtitle.Enable = FALSE;
	saturationSubtitle.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationSubtitle));

	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionSubtitle;
	noiseReductionSubtitle.Enable = FALSE;
	noiseReductionSubtitle.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionSubtitle));

	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementSubtitle;
	edgeEnhancementSubtitle.Enable = FALSE;
	edgeEnhancementSubtitle.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementSubtitle));

	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingSubtitle;
	anamorphicScalingSubtitle.Enable = FALSE;
	anamorphicScalingSubtitle.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingSubtitle));

	dwNumSurfs++;
#endif

#if 1
// BITMAP BACKGROUND
	DXVAHD_STREAM_STATE_D3DFORMAT_DATA d3dFormatBitmapBg;
	d3dFormatBitmapBg.Format = lw12Format;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_D3DFORMAT,
		sizeof(DXVAHD_STREAM_STATE_D3DFORMAT_DATA),
		(void *)(&d3dFormatBitmapBg));

	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA frameFormatBitmapBg;
	frameFormatBitmapBg.FrameFormat = DXVAHD_FRAME_FORMAT_PROGRESSIVE;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FRAME_FORMAT,
		sizeof(DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA),
		(void *)(&frameFormatBitmapBg));


	DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA inputColorSpaceDataBitmapBg;
    inputColorSpaceDataBitmapBg.RGB_Range = 1;
    inputColorSpaceDataBitmapBg.YCbCr_Matrix = 0;  // 0:BT.601(SDTV), 1:BT.709(HDTV)
    inputColorSpaceDataBitmapBg.YCbCr_xvYCC = 0;  // 0:Colwentional, 1:Expanded(xvYCC)
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE,
		sizeof(DXVAHD_STREAM_STATE_INPUT_COLOR_SPACE_DATA),
		(void *)(&inputColorSpaceDataBitmapBg));

	DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA outputRateBitmapBg;
	outputRateBitmapBg.RepeatFrame = FALSE;
	outputRateBitmapBg.OutputRate = DXVAHD_OUTPUT_RATE_NORMAL;
	outputRateBitmapBg.LwstomRate.Denominator = 1;
	outputRateBitmapBg.LwstomRate.Numerator = 1;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_OUTPUT_RATE,
		sizeof(DXVAHD_STREAM_STATE_OUTPUT_RATE_DATA),
		(void *)(&outputRateBitmapBg));

	DXVAHD_STREAM_STATE_SOURCE_RECT_DATA srcRectBitmapBg;
	srcRectBitmapBg.Enable = TRUE;
	//SetRect(&(srcRectBitmapBg.SourceRect), 0, 0, cxHDVideo, cyHDVideo);
	SetRect(&(srcRectBitmapBg.SourceRect), 0, 0, 0, 0);
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_SOURCE_RECT,
		sizeof(DXVAHD_STREAM_STATE_SOURCE_RECT_DATA),
		(void *)(&srcRectBitmapBg));

	DXVAHD_STREAM_STATE_ALPHA_DATA alphaBitmapBg;
	alphaBitmapBg.Enable = TRUE;
	alphaBitmapBg.Alpha = 0.;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaBitmapBg));

	DXVAHD_STREAM_STATE_PALETTE_DATA paletteBitmapBg;
	paletteBitmapBg.Count = 0;
#if 0
	paletteBitmapBg.pEntries = (D3DCOLOR *)calloc(paletteBitmapBg.Count, sizeof(D3DCOLOR));
	for (i = 0; i < paletteBitmapBg.Count; i++)
	{
		paletteBitmapBg.pEntries[i] = D3DCOLOR_ARGB(255, 255, 0, 0);
	}
#endif
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_PALETTE,
		sizeof(DXVAHD_STREAM_STATE_PALETTE_DATA),
		(void *)(&paletteBitmapBg));

	DXVAHD_STREAM_STATE_CLEAR_RECT_DATA clearRectBitmapBg;
	clearRectBitmapBg.ClearRectMask = 0;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_CLEAR_RECT,
		sizeof(DXVAHD_STREAM_STATE_CLEAR_RECT_DATA),
		(void *)(&clearRectBitmapBg));

	DXVAHD_STREAM_STATE_LUMA_KEY_DATA lumaKeyBitmapBg;
	lumaKeyBitmapBg.Enable = FALSE;
	lumaKeyBitmapBg.Lower = 0.;
	lumaKeyBitmapBg.Upper = 0.;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_LUMA_KEY,
		sizeof(DXVAHD_STREAM_STATE_LUMA_KEY_DATA),
		(void *)(&lumaKeyBitmapBg));

	DXVAHD_STREAM_STATE_FILTER_DATA brightnessBitmapBg;
	brightnessBitmapBg.Enable = FALSE;
	brightnessBitmapBg.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessBitmapBg));

	DXVAHD_STREAM_STATE_FILTER_DATA contrastBitmapBg;
	contrastBitmapBg.Enable = FALSE;
	contrastBitmapBg.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastBitmapBg));

	DXVAHD_STREAM_STATE_FILTER_DATA hueBitmapBg;
	hueBitmapBg.Enable = FALSE;
	hueBitmapBg.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueBitmapBg));

	DXVAHD_STREAM_STATE_FILTER_DATA saturationBitmapBg;
	saturationBitmapBg.Enable = FALSE;
	saturationBitmapBg.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationBitmapBg));

	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionBitmapBg;
	noiseReductionBitmapBg.Enable = FALSE;
	noiseReductionBitmapBg.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionBitmapBg));

	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementBitmapBg;
	edgeEnhancementBitmapBg.Enable = FALSE;
	edgeEnhancementBitmapBg.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementBitmapBg));

	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingBitmapBg;
	anamorphicScalingBitmapBg.Enable = FALSE;
	anamorphicScalingBitmapBg.Level = 128;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingBitmapBg));

	dwNumSurfs++;
#endif

// BltStates
	DXVAHD_BLT_STATE_BACKGROUND_COLOR_DATA BackgroundColor;
#if 0
	BackgroundColor.YCbCr = TRUE;
	BackgroundColor.BackgroundColor.Y = .3;
	BackgroundColor.BackgroundColor.Cr = .0;
	BackgroundColor.BackgroundColor.Cb = .9;
	BackgroundColor.BackgroundColor.A = 1.0;
#else
	BackgroundColor.YCbCr = FALSE;
	BackgroundColor.BackgroundColor.R = .9;
	BackgroundColor.BackgroundColor.G = .9;
	BackgroundColor.BackgroundColor.B = .9;
	BackgroundColor.BackgroundColor.A = 1.0;
#endif
	(videoData->m_pHDVP)->SetVideoProcessBltState(
				DXVAHD_BLT_STATE_BACKGROUND_COLOR,
				sizeof(DXVAHD_BLT_STATE_BACKGROUND_COLOR_DATA),
				(void *)(&BackgroundColor));

	DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE_DATA outputColorSpace;
	outputColorSpace.Usage = 0;
    outputColorSpace.RGB_Range = 1;
    outputColorSpace.YCbCr_Matrix = 0;  // 0:BT.601(SDTV), 1:BT.709(HDTV)
    outputColorSpace.YCbCr_xvYCC = 0;  // 0:Colwentional, 1:Expanded(xvYCC)
	(videoData->m_pHDVP)->SetVideoProcessBltState(
				DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE,
				sizeof(DXVAHD_BLT_STATE_OUTPUT_COLOR_SPACE_DATA),
				(void *)(&outputColorSpace));

	DXVAHD_BLT_STATE_ALPHA_FILL_DATA alphaFill;
	alphaFill.StreamNumber = 0;
	alphaFill.Mode = DXVAHD_ALPHA_FILL_MODE_BACKGROUND;
	(videoData->m_pHDVP)->SetVideoProcessBltState(
				DXVAHD_BLT_STATE_ALPHA_FILL,
				sizeof(DXVAHD_BLT_STATE_ALPHA_FILL_DATA),
				(void *)(&alphaFill));

	DXVAHD_BLT_STATE_DOWNSAMPLE_DATA downSample;
#if 1
	downSample.Enable = FALSE;
	downSample.Size.cx = 800;
	downSample.Size.cy = 600;
#else
	downSample.Enable = TRUE;
	downSample.Size.cx = (TargetRect.TargetRect.right - TargetRect.TargetRect.left);
	downSample.Size.cy = (TargetRect.TargetRect.bottom - TargetRect.TargetRect.top);
#endif
	(videoData->m_pHDVP)->SetVideoProcessBltState(
				DXVAHD_BLT_STATE_DOWNSAMPLE,
				sizeof(_DXVAHD_BLT_STATE_DOWNSAMPLE_DATA),
				(void *)(&downSample));

	DXVAHD_BLT_STATE_CLEAR_RECT_DATA clearRect;
	clearRect.Enable = FALSE;
	SetRect(&(clearRect.ClearRect[0]), 200, 200, 300, 300);
	//SetRect(&(clearRect.ClearRect[0]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[1]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[2]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[3]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[4]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[5]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[6]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[7]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[8]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[9]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[10]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[11]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[12]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[13]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[14]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[15]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[16]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[17]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[18]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[19]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[20]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[21]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[22]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[23]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[24]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[25]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[26]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[27]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[28]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[29]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[30]), 0, 0, 0, 0);
	SetRect(&(clearRect.ClearRect[31]), 0, 0, 0, 0);
	(videoData->m_pHDVP)->SetVideoProcessBltState(
				DXVAHD_BLT_STATE_CLEAR_RECT,
				sizeof(_DXVAHD_BLT_STATE_CLEAR_RECT_DATA),
				(void *)(&clearRect));

	return 0;
}

HRESULT VideoProcessSingleStream(
    CVideoData* videoData,
    UINT lwrrFrame,
    RECT& rcDest
)
{
    // Set up the DXVA2_VideoSample structure to hold information
    // about the video frame.
	int i;
	DWORD dwNumSurfs = 0;
	DXVA2_Fixed32 alpha;
	alpha.Value = 1;
	alpha.Fraction = 0;
	DXVAHD_STREAM_DATA StreamData[7];

	for (i = 0; i < 7; i++)
		memset(&(StreamData[i].InputVideoSamples), 0, sizeof(DXVA2_VideoSample));

#if 1
// MAIN VIDEO
// DXVA2.0
	StreamData[dwNumSurfs].InputVideoSamples.Start = (__int64)lwrrFrame * (__int64)periodVideo * 10000;
    StreamData[dwNumSurfs].InputVideoSamples.End
		= StreamData[dwNumSurfs].InputVideoSamples.Start + ((__int64)periodVideo * 10000);
    StreamData[dwNumSurfs].InputVideoSamples.SampleFormat = gSampleFormat;
    StreamData[dwNumSurfs].InputVideoSamples.SrcSurface = videoData->m_pVideoMemMain;
	StreamData[dwNumSurfs].InputVideoSamples.SampleData |= COMPOSITION_LAYER_MAILWIDEO;
	StreamData[dwNumSurfs].InputVideoSamples.PlanarAlpha = alpha;

	// the source rectangle is the entire surface.
    SetRect(&StreamData[dwNumSurfs].InputVideoSamples.SrcRect, 0, 0, cxHDVideo, cyHDVideo);

    // The destination rectangle is inset from the edges.
    SetRect(&StreamData[dwNumSurfs].InputVideoSamples.DstRect, insetPixels, insetPixels,
            rcDest.right - insetPixels, rcDest.bottom - insetPixels);

// DXVAHD

	DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA dstRectMailwideo;
	dstRectMailwideo.Enable = TRUE;
	SetRect(&(dstRectMailwideo.DestinationRect), insetPixels, insetPixels,
            rcDest.right - insetPixels, rcDest.bottom - insetPixels);
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_DESTINATION_RECT,
		sizeof(DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA),
		(void *)(&dstRectMailwideo));
	dwNumSurfs++;
#endif

#if 1
// SECONDARY VIDEO
// DXVA2.0
	StreamData[dwNumSurfs].InputVideoSamples.Start = (__int64)lwrrFrame * (__int64)periodVideo * 10000;
    StreamData[dwNumSurfs].InputVideoSamples.End
		= StreamData[dwNumSurfs].InputVideoSamples.Start + ((__int64)periodVideo * 10000);
    StreamData[dwNumSurfs].InputVideoSamples.SampleFormat = gSampleSSFormat;
    StreamData[dwNumSurfs].InputVideoSamples.SrcSurface = videoData->m_pVideoMemSecondary;
	StreamData[dwNumSurfs].InputVideoSamples.SampleData |= COMPOSITION_LAYER_SECONDARYVIDEO;

	// the source rectangle is the entire surface.
    SetRect(&StreamData[dwNumSurfs].InputVideoSamples.SrcRect, 0, 0, cxVideo, cyVideo);

    // The destination rectangle is inset from the edges.
    SetRect(&StreamData[dwNumSurfs].InputVideoSamples.DstRect, insetPixels, insetPixels,
            (cxVideo), (cyVideo));

// DXVAHD
	DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA dstRectSubVideo;
	dstRectSubVideo.Enable = TRUE;
	SetRect(&(dstRectSubVideo.DestinationRect), insetPixels, insetPixels, (cxVideo), (cyVideo));
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_DESTINATION_RECT,
		sizeof(DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA),
		(void *)(&dstRectSubVideo));

	dwNumSurfs++;
#endif

#if 1
// GRAPHICS
//DXVA2.0
    StreamData[dwNumSurfs].InputVideoSamples.Start = (__int64)lwrrFrame * (__int64)periodVideo * 10000;
    StreamData[dwNumSurfs].InputVideoSamples.End 
		= StreamData[dwNumSurfs].InputVideoSamples.Start + ((__int64)periodVideo * 10000);
    StreamData[dwNumSurfs].InputVideoSamples.SampleFormat = gSampleSSFormat;
    StreamData[dwNumSurfs].InputVideoSamples.SrcSurface = videoData->m_pVideoMemGraphics;
	StreamData[dwNumSurfs].InputVideoSamples.SampleData |= COMPOSITION_LAYER_GRAPHICS;

    // the source rectangle is the entire surface.
    SetRect(&StreamData[dwNumSurfs].InputVideoSamples.SrcRect, 0, 0, cxHDVideo, cyHDVideo);

    // The destination rectangle is inset from the edges.
    SetRect(&StreamData[dwNumSurfs].InputVideoSamples.DstRect, 0, rcDest.bottom*4/5,
            (rcDest.right), (rcDest.bottom)*9/10);

// DXVAHD
	DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA dstRectGraphics;
	dstRectGraphics.Enable = TRUE;
	SetRect(&(dstRectGraphics.DestinationRect), 0, rcDest.bottom * 4/5, (rcDest.right), (rcDest.bottom)*9/10);
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_DESTINATION_RECT,
		sizeof(DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA),
		(void *)(&dstRectGraphics));

	dwNumSurfs++;
#endif

#if 1
// SUBTITLE
	StreamData[dwNumSurfs].InputVideoSamples.Start = (__int64)lwrrFrame * (__int64)periodVideo * 10000;
    StreamData[dwNumSurfs].InputVideoSamples.End 
		= StreamData[dwNumSurfs].InputVideoSamples.Start + ((__int64)periodVideo * 10000);
    StreamData[dwNumSurfs].InputVideoSamples.SampleFormat = gSampleSSFormat;
	StreamData[dwNumSurfs].InputVideoSamples.SrcSurface = videoData->m_pVideoMemSubtitle; //reinterpret_cast<IDirect3DSurface9 *>(&gSampleSSFormat); //NULL; //videoData->m_pVideoMemGraphics;
	StreamData[dwNumSurfs].InputVideoSamples.SampleData |= COMPOSITION_LAYER_SUBTITLE;

// DXVAHD
	DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA dstRectSubtitle;
	dstRectSubtitle.Enable = TRUE;
	SetRect(&(dstRectSubtitle.DestinationRect), 0, rcDest.bottom*9/10, (rcDest.right), (rcDest.bottom));
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_DESTINATION_RECT,
		sizeof(DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA),
		(void *)(&dstRectSubtitle));

	dwNumSurfs++;
#endif

#if 1
// BITMAP BACKGROUND
    StreamData[dwNumSurfs].InputVideoSamples.Start = (__int64)lwrrFrame * (__int64)periodVideo * 10000;
    StreamData[dwNumSurfs].InputVideoSamples.End = StreamData[dwNumSurfs].InputVideoSamples.Start + ((__int64)periodVideo * 10000);
    StreamData[dwNumSurfs].InputVideoSamples.SampleFormat = gSampleSSFormat;
	StreamData[dwNumSurfs].InputVideoSamples.SrcSurface = videoData->m_pVideoMemBackground;
	StreamData[dwNumSurfs].InputVideoSamples.SampleData |= COMPOSITION_LAYER_BITMAP_BACKGROUND;

    // the source rectangle is the entire surface.
    SetRect(&StreamData[dwNumSurfs].InputVideoSamples.SrcRect, 0, 0, 1, 1);//35, 100, 55, 180);

    // The destination rectangle is inset from the edges.
	SetRect(&StreamData[dwNumSurfs].InputVideoSamples.DstRect, 0, 0, 1, 1);//100, 100, 180, 180);

// DXVAHD
	DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA dstRectBitmapBg;
	dstRectBitmapBg.Enable = TRUE;
	//SetRect(&(dstRectBitmapBg.DestinationRect), 0, rcDest.bottom*4/5, (rcDest.right), (rcDest.bottom));
	SetRect(&(dstRectBitmapBg.DestinationRect), 0, 0, 0, 0);
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_DESTINATION_RECT,
		sizeof(DXVAHD_STREAM_STATE_DESTINATION_RECT_DATA),
		(void *)(&dstRectBitmapBg));

	DXVAHD_STREAM_STATE_ALPHA_DATA alphaBitmapBg;
	alphaBitmapBg.Enable = TRUE;
	alphaBitmapBg.Alpha = 0.;
	videoData->m_pHDVP->SetVideoProcessStreamState(
		dwNumSurfs,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaBitmapBg));

	dwNumSurfs++;
#endif

    // Use a blue background color.
    const DXVA2_AYUVSample16 bkg = {0x00, 0xff00, 0xff00, 0xFFFF};

    // The video stream is 100% opaque
	const DXVA2_Fixed32 Alpha = {0, 1};

// DXVA2.0
	//(videoData->m_pHDVP)->VideoProcessBltParams = {0};
	(videoData->m_pHDVP)->VideoProcessBltParams.TargetFrame
		= StreamData[0].InputVideoSamples.Start;
	(videoData->m_pHDVP)->VideoProcessBltParams.TargetRect  = rcDest;
	(videoData->m_pHDVP)->VideoProcessBltParams.ConstrictionSize.cx = rcDest.right - rcDest.left;
	(videoData->m_pHDVP)->VideoProcessBltParams.ConstrictionSize.cy = rcDest.bottom - rcDest.top;
	(videoData->m_pHDVP)->VideoProcessBltParams.BackgroundColor = bkg;
	(videoData->m_pHDVP)->VideoProcessBltParams.DestFormat = gDestFormat;
	(videoData->m_pHDVP)->VideoProcessBltParams.ProcAmpValues.Brightness = gBrightness;
	(videoData->m_pHDVP)->VideoProcessBltParams.ProcAmpValues.Contrast = gContrast;
	(videoData->m_pHDVP)->VideoProcessBltParams.ProcAmpValues.Hue = gHue;
	(videoData->m_pHDVP)->VideoProcessBltParams.ProcAmpValues.Saturation = gSaturation;
	(videoData->m_pHDVP)->VideoProcessBltParams.Alpha = Alpha;

// DXVAHD
	DXVAHD_BLT_STATE_TARGET_RECT_DATA TargetRect;
	TargetRect.Enable = TRUE;
	TargetRect.TargetRect = rcDest;
	(videoData->m_pHDVP)->SetVideoProcessBltState(
				DXVAHD_BLT_STATE_TARGET_RECT,
				sizeof(DXVAHD_BLT_STATE_TARGET_RECT_DATA),
				(void *)(&TargetRect));

    // Blit the frame to the back buffer surface.
    HRESULT hr = videoData->m_pHDVP->VideoProcessBltHD(
                videoData->m_pD3DRt,
				0,
				dwNumSurfs,
				&StreamData[0]);

	return hr;
}

/******************************Public*Routine******************************\
* VideoProcessorIdle
*
* Draw the current video frame. 
*
\**************************************************************************/
UINT
VideoProcessorIdle(
    CVideoData* videoData,
    UINT timeNow
    )
{
    UINT lwrrFrame = (timeNow - videoData->m_startTime + periodVideo2) / periodVideo;

    //
    // generate a frame of video
    //
    PaintMailwideo(videoData->m_pVideoMemMain, lwrrFrame, cxHDVideo, cyHDVideo, videoData);
	
	PaintSubVideo(videoData->m_pVideoMemSecondary, lwrrFrame, cxVideo, cyVideo, videoData);

	PaintGraphics(videoData->m_pVideoMemGraphics, lwrrFrame, cxHDVideo, cyHDVideo);

	PaintSubtitle(videoData->m_pVideoMemSubtitle, lwrrFrame, cxHDVideo, cyHDVideo);

	PaintBitmapBackground(videoData->m_pVideoMemBackground, lwrrFrame, cxHDVideo, cyHDVideo);
	
	//
    // process the frame:
    //      - color space colwert it
    //      - scale to fit the client rectangle inset by 10 pixels
    //      - fill the inset region with background color (black)
    //
    RECT rcClient;
    GetClientRect(hwndApp,&rcClient);
	HRESULT hr = VideoProcessSingleStream(videoData, lwrrFrame, rcClient);

	//
    // Draw the frame to the screen.
    //
    hr = videoData->m_pD3DevEx->Present(&rcClient, &rcClient, NULL, NULL);
    if (hr == D3DERR_DEVICELOST)
    {
        // The h/w device is lost, can we get it back?
        hr = videoData->m_pD3DevEx->TestCooperativeLevel();
        if (hr == D3DERR_DEVICENOTRESET)
        {
            // TestCooperativeLevel returns DEVICENOTRESET when the h/w can be
            // recovered.
            CleanUpVideoResources(videoData);
            if (FALSE == CreateVideoResources(videoData, hwndApp))
            {
                // If we failed to recover, quit.
                PostQuitMessage(0);
            }
        }
    }
    else if (hr != S_OK)
    {
        // For any other failure, quit.
        PostQuitMessage(0);
    }
    //
    // callwlate how long to wait for the next frame - this callwlation will
    // overflow eventually.
    //
    timeNow = timeGetTime();
    UINT timeNext = videoData->m_startTime + (lwrrFrame + 1) * periodVideo;
    UINT timeWait = 0;

    if (timeNext < timeNow)
    {
        timeWait = periodVideo;
    }
    else
    {
        timeWait = timeNext - timeNow;
    }
    // wait for the next frame;
    return timeWait;
}


// CdxvahdmfcView construction/destruction

CdxvahdmfcView::CdxvahdmfcView()
{
	// TODO: add construction code here

}

CdxvahdmfcView::~CdxvahdmfcView()
{
}

BOOL CdxvahdmfcView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CdxvahdmfcView drawing

void CdxvahdmfcView::OnDraw(CDC* /*pDC*/)
{
	CdxvahdmfcDoc* pDoc = GetDolwment();
    //CVideoData* videoData = (CVideoData*)GetWindowLongPtr(m_hWnd, GWLP_USERDATA);
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
	VideoProcessorIdle(&gVideoData, timeGetTime());
}


// CdxvahdmfcView printing

BOOL CdxvahdmfcView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CdxvahdmfcView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CdxvahdmfcView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// CdxvahdmfcView diagnostics

#ifdef _DEBUG
void CdxvahdmfcView::AssertValid() const
{
	CView::AssertValid();
}

void CdxvahdmfcView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CdxvahdmfcDoc* CdxvahdmfcView::GetDolwment() const // non-debug version is inline
{
	ASSERT(m_pDolwment->IsKindOf(RUNTIME_CLASS(CdxvahdmfcDoc)));
	return (CdxvahdmfcDoc*)m_pDolwment;
}
#endif //_DEBUG


// CdxvahdmfcView message handlers

int CdxvahdmfcView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  Add your specialized creation code here
	//CVideoData* videoData = (CVideoData*)lpCreateStruct->lpCreateParams;
    //SetWindowLongPtr(m_hWnd, GWLP_USERDATA, (LONG_PTR)videoData);

    // Create the video processing resources.
	CreateVideoResources(&gVideoData, m_hWnd);
	hwndApp = m_hWnd;
	FrameRate = 25;
	SetTimer(0, 1000 /FrameRate , NULL);
	return 0;
}

void CdxvahdmfcView::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default
	Ilwalidate(FALSE);
	CView::OnTimer(nIDEvent);
}

void CdxvahdmfcView::OnEditConfig()
{
	// TODO: Add your command handler code here
	CConfigDlg dlg;
	// set the value of stream states
	// main video
	// Frame Format
	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA frameFormatMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FRAME_FORMAT,
		sizeof(DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA),
		(void *)(&frameFormatMailwideo));
	if (frameFormatMailwideo.FrameFormat == DXVAHD_FRAME_FORMAT_PROGRESSIVE)
		dlg.MVFrameFormat = FALSE;
	else
		dlg.MVFrameFormat = TRUE;

	// Alpha
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaMailwideo));
	dlg.MVAlphaEnable = alphaMailwideo.Enable;
	dlg.MVAlphaLevel = alphaMailwideo.Alpha * 100.;

	DXVAHD_STREAM_STATE_LUMA_KEY_DATA lumaKeyMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_LUMA_KEY,
		sizeof(DXVAHD_STREAM_STATE_LUMA_KEY_DATA),
		(void *)(&lumaKeyMailwideo));
	dlg.MVLumaKeyEnable = lumaKeyMailwideo.Enable;
	dlg.MVLumaKeyLower = lumaKeyMailwideo.Lower;
	dlg.MVLumakeyUpper = lumaKeyMailwideo.Upper;

	// Brightness
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessMailwideo));

	dlg.MVBrightnessEnable = brightnessMailwideo.Enable;
	dlg.MVBrightnessLevel = brightnessMailwideo.Level * 100 / 256;

	// Contrast
	DXVAHD_STREAM_STATE_FILTER_DATA contrastMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastMailwideo));
	dlg.MVContrastEnable = contrastMailwideo.Enable;
	dlg.MVContrastLevel = contrastMailwideo.Level * 100 / 256;

	// Hue
	DXVAHD_STREAM_STATE_FILTER_DATA hueMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueMailwideo));
	dlg.MVHueEnable = hueMailwideo.Enable;
	dlg.MVHueLevel = hueMailwideo.Level * 100 / 256;

	DXVAHD_STREAM_STATE_FILTER_DATA saturationMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationMailwideo));
	dlg.MVSaturationEnable = saturationMailwideo.Enable;
	dlg.MVSaturationLevel = saturationMailwideo.Level * 100 / 256;

	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionMailwideo));
	dlg.MVNoiseReductionEnable = noiseReductionMailwideo.Enable;
	dlg.MVNoiseReductionLevel = noiseReductionMailwideo.Level;

	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementMailwideo));
	dlg.MVEdgeEnhancementEnable = edgeEnhancementMailwideo.Enable;
	dlg.MVEdgeEnhancementLevel = edgeEnhancementMailwideo.Level * 100 / 256;

	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingMailwideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		0,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingMailwideo));
	dlg.MVAnamorphicScalingEnable= anamorphicScalingMailwideo.Enable;
	dlg.MVAnamorphicScalingLevel = anamorphicScalingMailwideo.Level * 100 / 256;


	// sub video
	// Frame Format
	DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA frameFormatSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FRAME_FORMAT,
		sizeof(DXVAHD_STREAM_STATE_FRAME_FORMAT_DATA),
		(void *)(&frameFormatSubVideo));
	if (frameFormatSubVideo.FrameFormat == DXVAHD_FRAME_FORMAT_PROGRESSIVE)
		dlg.SVFrameFormat = FALSE;
	else
		dlg.SVFrameFormat = TRUE;

	// Alpha
	DXVAHD_STREAM_STATE_ALPHA_DATA alphaSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_ALPHA,
		sizeof(DXVAHD_STREAM_STATE_ALPHA_DATA),
		(void *)(&alphaSubVideo));
	dlg.SVAlphaEnable = alphaSubVideo.Enable;
	dlg.SVAlphaLevel = alphaSubVideo.Alpha * 100.;

	DXVAHD_STREAM_STATE_LUMA_KEY_DATA lumaKeySubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_LUMA_KEY,
		sizeof(DXVAHD_STREAM_STATE_LUMA_KEY_DATA),
		(void *)(&lumaKeySubVideo));
	dlg.SVLumaKeyEnable = lumaKeySubVideo.Enable;
	dlg.SVLumaKeyLower = lumaKeySubVideo.Lower;
	dlg.SVLumaKeyUpper = lumaKeySubVideo.Upper;

	// Brightness
	DXVAHD_STREAM_STATE_FILTER_DATA brightnessSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_BRIGHTNESS,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&brightnessSubVideo));

	dlg.SVBrightnessEnable = brightnessSubVideo.Enable;
	dlg.SVBrightnessLevel = brightnessSubVideo.Level * 100 / 256;

	// Contrast
	DXVAHD_STREAM_STATE_FILTER_DATA contrastSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_CONTRAST,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&contrastSubVideo));
	dlg.SVContrastEnable = contrastSubVideo.Enable;
	dlg.SVContrastLevel = contrastSubVideo.Level * 100 / 256;

	// Hue
	DXVAHD_STREAM_STATE_FILTER_DATA hueSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_HUE,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&hueSubVideo));
	dlg.SVHueEnable = hueSubVideo.Enable;
	dlg.SVHueLevel = hueSubVideo.Level * 100 / 256;

	DXVAHD_STREAM_STATE_FILTER_DATA saturationSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_SATURATION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&saturationSubVideo));
	dlg.SVSaturationEnable = saturationSubVideo.Enable;
	dlg.SVSaturationlevel = saturationSubVideo.Level * 100 / 256;

	DXVAHD_STREAM_STATE_FILTER_DATA noiseReductionSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_NOISE_REDUCTION,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&noiseReductionSubVideo));
	dlg.SVNoiseReductionEnable = noiseReductionSubVideo.Enable;
	dlg.SVNoiseReductionLevel = noiseReductionSubVideo.Level;

	DXVAHD_STREAM_STATE_FILTER_DATA edgeEnhancementSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_EDGE_ENHANCEMENT,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&edgeEnhancementSubVideo));
	dlg.SVEdgeEnhancementEnable = edgeEnhancementSubVideo.Enable;
	dlg.SVEdgeENhancementLevel = edgeEnhancementSubVideo.Level * 100 / 256;

	DXVAHD_STREAM_STATE_FILTER_DATA anamorphicScalingSubVideo;
	gVideoData.m_pHDVP->GetVideoProcessStreamState(
		1,
		DXVAHD_STREAM_STATE_FILTER_ANAMORPHIC_SCALING,
		sizeof(DXVAHD_STREAM_STATE_FILTER_DATA),
		(void *)(&anamorphicScalingSubVideo));
	dlg.SVAnamorphicScalingEnable= anamorphicScalingSubVideo.Enable;
	dlg.SVAnamorphicScalingLevel = anamorphicScalingSubVideo.Level * 100 / 256;

	// graphics
	// subtitle
	// bitmap background

	// blt states
	DXVAHD_BLT_STATE_TARGET_RECT_DATA TargetRect;
	gVideoData.m_pHDVP->GetVideoProcessBltState(
				DXVAHD_BLT_STATE_TARGET_RECT,
				sizeof(DXVAHD_BLT_STATE_TARGET_RECT_DATA),
				(void *)(&TargetRect));

	DXVAHD_BLT_STATE_DOWNSAMPLE_DATA downSample;
	gVideoData.m_pHDVP->GetVideoProcessBltState(
				DXVAHD_BLT_STATE_DOWNSAMPLE,
				sizeof(_DXVAHD_BLT_STATE_DOWNSAMPLE_DATA),
				(void *)(&downSample));
	dlg.BltDownSampleEnable = downSample.Enable;
	dlg.BltDownSampleLevel = 100 * downSample.Size.cx
		/ (TargetRect.TargetRect.right - TargetRect.TargetRect.left);

	if (dlg.DoModal() == IDOK)
	{
	}

}

void CdxvahdmfcView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: Add your message handler code here and/or call default
	switch (nChar)
	{
	case VK_UP:
		if (FrameRate < 25)
			FrameRate++;
		SetTimer(0, 1000 /FrameRate , NULL);
		break;
	case VK_DOWN:
		if (FrameRate > 5)
			FrameRate--;
		SetTimer(0, 1000 /FrameRate , NULL);
		break;
	}
	CView::OnKeyDown(nChar, nRepCnt, nFlags);
}
