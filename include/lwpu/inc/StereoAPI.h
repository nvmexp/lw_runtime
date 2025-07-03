 /***************************************************************************\
|*                                                                           *|
|*        Copyright (c) 1993-2000 LWPU, Corp.  All rights reserved.        *|
|*                                                                           *|
|*     NOTICE TO USER:   The source code  is copyrighted under  U.S. and     *|
|*     international laws.   LWPU, Corp. of Sunnyvale, California owns     *|
|*     the copyright  and as design patents  pending  on the design  and     *|
|*     interface  of the LW chips.   Users and possessors of this source     *|
|*     code are hereby granted  a nonexclusive,  royalty-free  copyright     *|
|*     and  design  patent license  to use this code  in individual  and     *|
|*     commercial software.                                                  *|
|*                                                                           *|
|*     Any use of this source code must include,  in the user dolwmenta-     *|
|*     tion and  internal comments to the code,  notices to the end user     *|
|*     as follows:                                                           *|
|*                                                                           *|
|*     Copyright (c) 1993-2000  LWPU, Corp.    LWPU  design  patents     *|
|*     pending in the U.S. and foreign countries.                            *|
|*                                                                           *|
|*     LWPU, CORP.  MAKES  NO REPRESENTATION ABOUT  THE SUITABILITY OF     *|
|*     THIS SOURCE CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT     *|
|*     EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORP. DISCLAIMS     *|
|*     ALL WARRANTIES  WITH REGARD  TO THIS SOURCE CODE,  INCLUDING  ALL     *|
|*     IMPLIED   WARRANTIES  OF  MERCHANTABILITY  AND   FITNESS   FOR  A     *|
|*     PARTICULAR  PURPOSE.   IN NO EVENT SHALL LWPU, CORP.  BE LIABLE     *|
|*     FOR ANY SPECIAL, INDIRECT, INCIDENTAL,  OR CONSEQUENTIAL DAMAGES,     *|
|*     OR ANY DAMAGES  WHATSOEVER  RESULTING  FROM LOSS OF USE,  DATA OR     *|
|*     PROFITS,  WHETHER IN AN ACTION  OF CONTRACT,  NEGLIGENCE OR OTHER     *|
|*     TORTIOUS ACTION, ARISING OUT  OF OR IN CONNECTION WITH THE USE OR     *|
|*     PERFORMANCE OF THIS SOURCE CODE.                                      *|
|*                                                                           *|
 \***************************************************************************/

/******************* Stereo API Defines and Structures *********************\
*                                                                           *
* Module: StereoAPI.h                                                       *
*                                                                           *
\***************************************************************************/

#ifndef STEREO_API_H
#define STEREO_API_H

#ifdef  DEFINE_API_INTERNALS
#define DLLTYPE __declspec( dllexport )
#else
#define DLLTYPE __declspec( dllimport )
#endif


// StereoAPI interface version has to be used with CheckAPIVersion
#define STEREOAPI_VERSION       1

// Stereo Lock Control codes used as parameters to the EyeAccessControl API. This method controls
// subsequent calls to the DX "lock surface" API. Default setting is DEFAULT_BUFFER and if stereo
// is active forces LOCK call to return an emulated color buffer.
#define DEFAULT_BUFFER  0
#define LEFT_BUFFER     1
#define RIGHT_BUFFER    2

// SetFrustumAdjustMode codes.
#define NO_ADJUST       0   //Preferable for games providing extra data clipped by the frustum. Default.
#define STRETCH         1   //Stretches the frustum in X to compensate for missing eye data.
#define CLEAR_EDGES     2   //Clears the missing eye data.

// CaptureImageFormats.
#define IMAGE_JPEG     0
#define IMAGE_PNG      1


// Stereo states: bit 0 - on/off; bit 1 - enabled/disabled.
// Returned by GetStereoState.
#define STEREO_STATE_ENABLED         0x2
#define STEREO_STATE_DISABLED        0x0
#define STEREO_STATE_ON              0x1
#define STEREO_STATE_OFF             0x0

// Image quality applies to JPEG only and varies from 0 to 100. 100 being the best.
// Default setting for CaptureImage is JPEG and quality=75. 

#if defined(__cplusplus) && !defined(IS_OPENGL)

interface IStereoAPI
{
    virtual HRESULT __stdcall       QueryInterface(void *riid, void **ppvObject) = 0;
    virtual ULONG   __stdcall       AddRef() = 0;
    virtual ULONG   __stdcall       Release() = 0;
    virtual int     __stdcall       StereoOn(void) = 0;
    virtual int     __stdcall       StereoOff(void) = 0;
    virtual float   __stdcall       GetSeparation(void) = 0;
    virtual float   __stdcall       SetSeparation(float) = 0;
    virtual float   __stdcall       IncreaseSeparation(void) = 0;
    virtual float   __stdcall       DecreaseSeparation(void) = 0;
    virtual void    __stdcall       EyeAccessControl(int) = 0;
    virtual float   __stdcall       GetColwergence(void) = 0;
    virtual float   __stdcall       SetColwergence(float) = 0;
    virtual float   __stdcall       IncreaseColwergence(void) = 0;
    virtual float   __stdcall       DecreaseColwergence(void) = 0;
    virtual int     __stdcall       GetFrustumAdjustMode(void) = 0;
    virtual int     __stdcall       SetFrustumAdjustMode(int) = 0;
    virtual int     __stdcall       OrthoStereoOn(void) = 0;
    virtual int     __stdcall       OrthoStereoOff(void) = 0;
    virtual float   __stdcall       GetGammaAdjustment(void) = 0;
    virtual float   __stdcall       SetGammaAdjustment(float) = 0;
    virtual float   __stdcall       IncreaseGammaAdjustment(void) = 0;
    virtual float   __stdcall       DecreaseGammaAdjustment(void) = 0;
    virtual void    __stdcall       CaptureImageFormat(int format, int quality) = 0;
    virtual void    __stdcall       CaptureImage(void) = 0;
    virtual BOOL    __stdcall       isValid(void) = 0;
    virtual int     __stdcall       SetStereoState(int) = 0;
    virtual ULONG   __stdcall       GetStereoState(void) = 0;
    virtual int     __stdcall       CheckAPIVersion(int) = 0;

};

typedef IStereoAPI CSTEREOAPI, *PCSTEREOAPI;


typedef int __cdecl CreateStereoAPIFunction(PCSTEREOAPI *pStereoAPI);
typedef int __cdecl IsStereoEnabledFunction(void);

//extern "C" __declspec( dllexport ) int __cdecl CreateStereoAPI(PCSTEREOAPI *pStereoAPI);
extern "C" __declspec( dllexport ) int __cdecl IsStereoEnabled(void);
extern "C" __declspec( dllexport ) int __cdecl CreateStereoAPI(PCSTEREOAPI *ppStereoAPI);

#else   //__cplusplus==0

struct  StereoAPIVtbl;

typedef struct _StereoAPI
{
    struct  StereoAPIVtbl  *lpVtbl;
} CSTEREOAPI, *PCSTEREOAPI;

// "C" interface structure for Stereo API.
// Function prototypes used in Vtbl
typedef long            (__stdcall *pfn1)(PCSTEREOAPI, void *, void **);
typedef unsigned long   (__stdcall *pfn2)(PCSTEREOAPI);
typedef int             (__stdcall *pfn3)(PCSTEREOAPI);
typedef float           (__stdcall *pfn4)(PCSTEREOAPI);
typedef float           (__stdcall *pfn5)(PCSTEREOAPI, float);
typedef void            (__stdcall *pfn6)(PCSTEREOAPI, int);
typedef int             (__stdcall *pfn7)(PCSTEREOAPI,int);
typedef void            (__stdcall *pfn8)(PCSTEREOAPI, int format, int quality);
typedef void            (__stdcall *pfn9)(PCSTEREOAPI);

struct  StereoAPIVtbl {
    pfn1            QueryInterface;
    pfn2            AddRef;
    pfn2            Release;
    pfn3            StereoOn;
    pfn3            StereoOff;
    pfn4            GetSeparation;
    pfn5            SetSeparation;
    pfn4            IncreaseSeparation;
    pfn4            DecreaseSeparation;
    pfn6            EyeAccessControl;
    pfn4            GetColwergence;
    pfn5            SetColwergence;
    pfn4            IncreaseColwergence;
    pfn4            DecreaseColwergence;
    pfn3            GetFrustumAdjustMode;
    pfn7            SetFrustumAdjustMode;
    pfn3            OrthoStereoOn;
    pfn3            OrthoStereoOff;
    pfn4            GetGammaAdjustment;
    pfn5            SetGammaAdjustment;
    pfn4            IncreaseGammaAdjustment;
    pfn4            DecreaseGammaAdjustment;
    pfn8            CaptureImageFormat;
    pfn9            CaptureImage;
    pfn3            isValid;
    pfn7            SetStereoState;
    pfn2            GetStereoState;
    pfn7            CheckAPIVersion;
/*
    long            (__stdcall *QueryInterface)(PCSTEREOAPI, void *, void **);
    unsigned long   (__stdcall *AddRef)(PCSTEREOAPI);
    unsigned long   (__stdcall *Release)(PCSTEREOAPI);
    int             (__stdcall *StereoOn)(PCSTEREOAPI);
    int             (__stdcall *StereoOff)(PCSTEREOAPI);
    float           (__stdcall *GetSeparation)(PCSTEREOAPI);
    float           (__stdcall *SetSeparation)(PCSTEREOAPI, float);
    float           (__stdcall *IncreaseSeparation)(PCSTEREOAPI);
    float           (__stdcall *DecreaseSeparation)(PCSTEREOAPI);
    void            (__stdcall *EyeAccessControl)(PCSTEREOAPI, int);
    float           (__stdcall *GetColwergence)(PCSTEREOAPI);
    float           (__stdcall *SetColwergence)(PCSTEREOAPI, float);
    float           (__stdcall *IncreaseColwergence)(PCSTEREOAPI);
    float           (__stdcall *DecreaseColwergence)(PCSTEREOAPI);
    int             (__stdcall *GetFrustumAdjustMode)(PCSTEREOAPI);
    int             (__stdcall *SetFrustumAdjustMode)(PCSTEREOAPI,int);
    int             (__stdcall *OrthoStereoOn)(PCSTEREOAPI);
    int             (__stdcall *OrthoStereoOff)(PCSTEREOAPI);
    float           (__stdcall *GetGammaAdjustment)(PCSTEREOAPI);
    float           (__stdcall *SetGammaAdjustment)(PCSTEREOAPI,float);
    float           (__stdcall *IncreaseGammaAdjustment)(PCSTEREOAPI);
    float           (__stdcall *DecreaseGammaAdjustment)(PCSTEREOAPI);
    void            (__stdcall *CaptureImageFormat)(PCSTEREOAPI, int format, int quality);
    void            (__stdcall *CaptureImage)(PCSTEREOAPI);
    BOOL            (__stdcall *isValid)(PCSTEREOAPI);
    int             (__stdcall *SetStereoState)(PCSTEREOAPI, int);
    ULONG           (__stdcall *GetStereoState)(PCSTEREOAPI);
    int             (__stdcall *CheckAPIVersion)(PCSTEREOAPI, int);
*/
};


typedef int __cdecl CreateStereoAPIFunction(PCSTEREOAPI *pStereoAPI);
typedef int __cdecl IsStereoEnabledFunction(void);

extern __declspec( dllexport ) int __cdecl IsStereoEnabled(void);
extern __declspec( dllexport ) int __cdecl CreateStereoAPI(PCSTEREOAPI *ppStereoAPI);



#endif  //__cplusplus==0

/* This is a sample of the dynamic StereoAPI creation
int DynCreateStereoAPI(PCSTEREOAPI *pStereoAPI)
{
    HMODULE Lib;
    if (Lib=LoadLibraryA("Stereoi.dll"))
    {
    CreateStereoAPIFunction* CreateStereoAPI = NULL;
        if (CreateStereoAPI=(CreateStereoAPIFunction*)GetProcAddress (Lib,"CreateStereoAPI"))
            return CreateStereoAPI(pStereoAPI);
    }
    return 0;
}
*/

// Stereo Blit defines.

#define LWSTEREO_IMAGE_SIGNATURE 0x4433564e //LW3D

typedef struct  _Lw_Stereo_Image_Header
{
    unsigned int    dwSignature;
    unsigned int    dwWidth;
    unsigned int    dwHeight;
    unsigned int    dwBPP;
    unsigned int    dwFlags;
} LWSTEREOIMAGEHEADER, *LPLWSTEREOIMAGEHEADER;

// ORed flags in the dwFlags fiels of the _Lw_Stereo_Image_Header structure above 
#define     SIH_SWAP_EYES               0x00000001
#define     SIH_SCALE_TO_FIT            0x00000002

#endif //STEREO_API_H

