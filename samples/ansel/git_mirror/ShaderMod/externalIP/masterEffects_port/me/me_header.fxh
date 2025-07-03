#include "MasterEffect.h"

//global vars

#define BUFFER_WIDTH	(float)(screenSize.x)
#define BUFFER_HEIGHT	(float)(screenSize.y)
#define BUFFER_RCP_WIDTH	1/BUFFER_WIDTH
#define BUFFER_RCP_HEIGHT	1/BUFFER_HEIGHT

#define ScreenSize 	float4(BUFFER_WIDTH, BUFFER_RCP_WIDTH, float(BUFFER_WIDTH) / float(BUFFER_HEIGHT), float(BUFFER_HEIGHT) / float(BUFFER_WIDTH)) //x=Width, y=1/Width, z=ScreenScaleY, w=1/ScreenScaleY
#define PixelSize  	float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)
#define PI 		3.1415972
#define PIOVER180 	0.017453292
#define AUTHOR 		MartyMcFly
#define FOV 		75
#define LumCoeff 	float3(0.212656, 0.715158, 0.072186)
#define zFarPlane 	1
#define zNearPlane 	0.001		//I know, weird values but ReShade's depthbuffer is ... odd
#define aspect          (BUFFER_RCP_HEIGHT/BUFFER_RCP_WIDTH)
#define IlwFocalLen 	float2(tan(0.5f*radians(FOV)) / (float)BUFFER_RCP_HEIGHT * (float)BUFFER_RCP_WIDTH, tan(0.5f*radians(FOV)))

//uniform float4 Timer < source = "timer"; >;

#if( USE_HDR_LEVEL == 0)
 #define RENDERMODE RGBA8
#endif
#if( USE_HDR_LEVEL == 1)
 #define RENDERMODE RGBA16F
#endif
#if( USE_HDR_LEVEL == 2)
 #define RENDERMODE RGBA32F
#endif
