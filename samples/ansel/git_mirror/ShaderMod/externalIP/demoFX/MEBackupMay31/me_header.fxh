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

//textures
/*
texture   texBloom1 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RENDERMODE;};
texture   texBloom2 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RENDERMODE;};
texture   texBloom3 { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RENDERMODE;};
texture   texBloom4 { Width = BUFFER_WIDTH/4; Height = BUFFER_HEIGHT/4; Format = RENDERMODE;};
texture   texBloom5 { Width = BUFFER_WIDTH/8; Height = BUFFER_HEIGHT/8; Format = RENDERMODE;};

texture   texLens1 { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RENDERMODE;};
texture   texLens2 { Width = BUFFER_WIDTH/2; Height = BUFFER_HEIGHT/2; Format = RENDERMODE;};

texture2D texLDR : COLOR;
texture   texHDR1 	{ Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; MipLevels = 5; Format = RENDERMODE;};	
texture   texHDR2 	{ Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; MipLevels = 5; Format = RENDERMODE;};

texture   texOcclusion1 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT;  Format = RGBA16F;}; //MUST be at least 16, 8 gives heavy artifacts when blurring.
texture   texOcclusion2 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT;  Format = RGBA16F;}; //"Optimizations" can be done elsewhere, not here.

texture   texCoC	{ Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT;  MipLevels = 5; Format = RGBA16F;};
texture2D texDepth : DEPTH;
-*/

/*
texture   texNoise  < string source = "MasterEffect/internal/mcnoise.png";  	> {Width = BUFFER_WIDTH;Height = BUFFER_HEIGHT;Format = RGBA8;};
texture   texSprite < string source = "MasterEffect/mcsprite.png"; 	 	> {Width = BUFFER_WIDTH;Height = BUFFER_HEIGHT;Format = RGBA8;};
texture   texDirt   < string source = "MasterEffect/mcdirt.png";   		> {Width = BUFFER_WIDTH;Height = BUFFER_HEIGHT;Format = RGBA8;};
texture   texLUT    < string source = "MasterEffect/mclut.png";    		> {Width = 256; Height = 1;   Format = RGBA8;};
texture   texLUT3D  < string source = "MasterEffect/mclut3d.png";  		> {Width = 256; Height = 16;   Format = RGBA8;};
texture   texMask   < string source = "MasterEffect/mcmask.png";   		> {Width = BUFFER_WIDTH;Height = BUFFER_HEIGHT;Format = R8;};
texture   texHeat   < string source = "MasterEffect/internal/mcheat.png";   	> {Width = 512;Height = 512;Format = RGBA8;};
*/