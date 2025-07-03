  /*------------------------------.
  | :: Include enabled shaders :: |
  '------------------------------*/


  /*----------------------.
  | ::   Shared Pass   :: |
  '----------------------*/

/* //didn't make it into 1.6
#if (USE_PRINT == 1) //I could make this always on
  #include "Print.h"
  #define USE_SHARED 1
#endif
*/

//didn't make it into 1.6	
/*
#if (USE_SWEETCRT == 1)
  #include "SweetCRT.h"
  #define USE_SHARED 1
#endif
*/

#if (USE_LEVELS == 1)
  #include "Levels.h"
  #define USE_SHARED 1
#endif

#if (USE_TECHNICOLOR == 1)
  #include "Technicolor.h"
  #define USE_SHARED 1
#endif

#if (USE_TECHNICOLOR2 == 1)
  #include "Technicolor2.h"
  #define USE_SHARED 1
#endif

#if (USE_DPX == 1)
  #include "DPX.h"
  #define USE_SHARED 1
#endif

#if (USE_MONOCHROME == 1)
  #include "Monochrome.h"
  #define USE_SHARED 1
#endif

#if (USE_COLORMATRIX == 1)
  #include "ColorMatrix.h"
  #define USE_SHARED 1
#endif

#if (USE_LIFTGAMMAGAIN == 1)
  #include "LiftGammaGain.h"
  #define USE_SHARED 1
#endif

#if (USE_TONEMAP == 1)
  #include "Tonemap.h"
  #define USE_SHARED 1
#endif

#if (USE_VIBRANCE == 1)
  #include "Vibrance.h"
  #define USE_SHARED 1
#endif

#if (USE_LWRVES == 1)
  #include "Lwrves.h"
  #define USE_SHARED 1
#endif

#if (USE_SEPIA == 1)
  #include "Sepia.h"
  #define USE_SHARED 1
#endif

/* //didn't make it into 1.6	
#if (USE_DALTONIZE == 1)
  #include "Daltonize.h"
  #define USE_SHARED 1
#endif
*/

#if (USE_NOSTALGIA == 1)
  #include "Nostalgia.h"
  #define USE_SHARED 1
#endif

#if (USE_VIGNETTE == 1)
  #include "Vignette.h"
  #define USE_SHARED 1
#endif

#if (USE_FILMGRAIN == 1)
  #include "FilmGrain.h"
  #define USE_SHARED 1
#endif

#if (USE_DITHER == 1)
  #include "Dither.h"
  #define USE_SHARED 1
#endif

#if (USE_BORDER == 1)
  #include "Border.h"
  #define USE_SHARED 1
#endif

#if (USE_SPLITSCREEN == 1)
  #include "Splitscreen.h"
  #define USE_SHARED 1
#endif

#if (USE_SHARED == 1)
  #include "Shared.h"
#endif


  /*----------------------------------.
  | :: Begin operation "Piggyback" :: |
  '----------------------------------*/
// Operation "Piggyback" is where we track what pass came before the shared pass,
// so it can piggyback on the previous pass instead of running in it's own -
// thus avoid the overhead of another pass and increasing performance.

// Test to see what was the last effect applied.
// To do this I test to see if each effect was applied in the reverse order they run in.
// Starting with the last and continueing to the second last if it was not enabled.

//It's important that this list is the exact reverse of the order they run in.

#if (USE_SHARED == 1) //only do this if we need one of the effects in the shared pass
	#if USE_LUMASHARPEN
		#define Shared_Piggyback_LumaSharpen 1
	#elif USE_HDR
		#define Shared_Piggyback_HDR 1
	#elif USE_BLOOM
		#define Shared_Piggyback_Bloom 1
	#elif USE_PIXELART_CRT
		#define Shared_Piggyback_PixelArt_CRT 1
	#elif USE_ADVANCED_CRT
		#define Shared_Piggyback_Advanced_CRT 1
	#elif USE_CA
		#define Shared_Piggyback_CA 1
	#elif USE_EXPLOSION
		#define Shared_Piggyback_Explosion
	#elif USE_FXAA
		#define Shared_Piggyback_FXAA 1
	#elif USE_SMAA 
		#define Shared_Piggyback_SMAA 1
	#elif USE_CARTOON
		#define Shared_Piggyback_Cartoon 1
	#elif USE_ASCII
		#define Shared_Piggyback_Ascii 1
	#else
		#define Shared_Piggyback_No 1 // No previous passes are running - we need to make one.
	#endif
#endif


  /*-----------------------.
  | ::      ASCII       :: |
  '-----------------------*/

#if (USE_ASCII == 1)
  // Ascii
  #include "Ascii.h"
#endif

  /*-----------------------.
  | ::     Cartoon      :: |
  '-----------------------*/

#if (USE_CARTOON == 1)
  // Cartoon
  #include "Cartoon.h"
#endif

  /*-----------------------.
  | ::   LumaSharpen    :: |
  '-----------------------*/
  
#if (USE_LUMASHARPEN == 1)
  // LumaSharpen
  #include "LumaSharpen.h"
#endif

  /*--------------------.
  | ::     SMAA      :: |
  '--------------------*/
  
  //TODO Move SMAA Wrappers to seperate file

#if (USE_SMAA == 1 || USE_SMAA_ANTIALIASING == 1)

	#define SMAA_RT_METRICS float4(pixel, screen_size) //let SMAA know the size of a pixel and the screen
  
  //#define SMAA_HLSL_3 1
  #define SMAA_LWSTOM_SL 1 //our own reshade branch
  
  #define SMAA_PIXEL_SIZE pixel
  #define SMAA_PRESET_LWSTOM 1

  #include "SMAA.hlsl"
  #include "SMAAWrap.h"
#endif

  /*--------------------.
  | ::     FXAA      :: |
  '--------------------*/

#if (USE_FXAA == 1 || USE_FXAA_ANTIALIASING == 1)

  #define FXAA_PC 1
  #define FXAA_HLSL_3 1
  #define FXAA_GREEN_AS_LUMA 1 //It's better to callwlate luma in the previous pass and pass it, than to use this option.

  #include "Fxaa3_11.h"
  #include "FXAAWrap.h"


#endif

  /*-----------------------.
  | ::      Bloom       :: |
  '-----------------------*/

#if (USE_BLOOM == 1)
  #include "Bloom.h"
#endif


  /*-----------------------.
  | ::       HDR        :: |
  '-----------------------*/

#if (USE_HDR == 1)
  #include "HDR.h"
#endif


  /*---------------------------.
  | :: Chromatic Aberration :: |
  '---------------------------*/

#if (USE_CA == 1)
  #include "ChromaticAberration.h"
#endif

  /*-----------------------.
  | ::     Explosion    :: |
  '-----------------------*/

#if (USE_EXPLOSION == 1)
  // Explosion
  #include "Explosion.h"
#endif

  /*-----------------------.
  | ::   Advanced CRT   :: |
  '-----------------------*/

#if (USE_ADVANCED_CRT == 1)
  // Advanced CRT
  #include "AdvancedCRT.h"
#endif

  /*-----------------------.
  | ::   PixelArt CRT   :: |
  '-----------------------*/

#if (USE_PIXELART_CRT == 1)
  #include "PixelArtCRT.h"
#endif

  /*------------------------.
  | ::  Lens Distortion  :: |
  '------------------------*/

#if (USE_LENS_DISTORTION == 1)
  #include "LensDistortion.h"
#endif


  /*-----------------------.
  | ::      Custom      :: |
  '-----------------------*/
#if (USE_LWSTOM == 1)
  #include "Custom.h"
#endif

  /*----------------------.
  | ::  Display Depth  :: |
  '----------------------*/

#if (USE_DEPTH == 1)
  #include "DisplayDepth.h"
#endif

  /*----------------------.
  | ::   Transitions   :: |
  '----------------------*/

#if (USE_TRANSITION == 1)
  #include "Transitions.h"
#endif
