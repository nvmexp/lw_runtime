/*=============================================================================

	ReShade 4 effect file
    github.com/martymcmodding

	Support me:
   		paypal.me/mcflypg
   		patreon.com/mcflypg

    Path Traced Global Illumination 

    by Marty McFly / P.Gilcher
    part of qUINT shader library for ReShade 4

    CC BY-NC-ND 3.0 licensed.

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

//these two are required if I want to reuse the MXAO textures properly

#ifndef MXAO_MIPLEVEL_AO
 #define MXAO_MIPLEVEL_AO		0	//[0 to 2]      Miplevel of AO texture. 0 = fullscreen, 1 = 1/2 screen width/height, 2 = 1/4 screen width/height and so forth. Best results: IL MipLevel = AO MipLevel + 2
#endif

#ifndef MXAO_MIPLEVEL_IL
 #define MXAO_MIPLEVEL_IL		2	//[0 to 4]      Miplevel of IL texture. 0 = fullscreen, 1 = 1/2 screen width/height, 2 = 1/4 screen width/height and so forth.
#endif

#ifndef INFINITE_BOUNCES
 #define INFINITE_BOUNCES       0   //[0 or 1]      If enabled, path tracer samples previous frame GI as well, causing a feedback loop to simulate secondary bounces, causing a more widespread GI.
#endif

#ifndef SPATIAL_FILTER
 #define SPATIAL_FILTER	       	1   //[0 or 1]      If enabled, final GI is filtered for a less noisy but also less precise result. Enabled by default.
#endif

#ifndef DEBUG_TESTING
 #define DEBUG_TESTING	       	0   //[0 or 1]      If enabled, Extra UI inputs will be available such as "Filter Active" and "Enable Debug View".
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/
/*
uniform float RT_SIZE_SCALE <
	ui_type = "drag";
	ui_min = 0.25; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "GI Render Resolution Scale";
    ui_category = "Global";
> = 1.0;
*/

#if DEBUG_TESTING != 0
	uniform bool RT_FILTER_ACTIVE
	<
		ui_label = "Filter Active";
		ui_tooltip = "Toggles filter on/off.";
		ui_category = "Blending";
	> = true;

	uniform bool RT_DEBUG_VIEW
	<
		ui_label = "Enable Debug View";
		ui_tooltip = "Different debug outputs";
		ui_category = "Debug";
	> = false;
#endif

uniform bool RT_ILWERT_DEPTH
<
    ui_label = "Ilwert Depth";
	ui_tooltip = "Ilwerts the depth value.";
    ui_category = "Blending";
> = false;

uniform bool RT_FLIP_DEPTH
<
    ui_label = "Flip Depth";
	ui_tooltip = "Flips the depth buffer vertically.";
    ui_category = "Blending";
> = false;

uniform float RT_SAMPLE_RADIUS <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 20.0;
    ui_step = 0.01;
    ui_label = "Ray Step Length";
	ui_tooltip = "Maximum ray length, directly affects\nthe spread radius of shadows / indirect lighing";
    ui_category = "Path Tracing";
> = 15.0;

uniform float RT_RAY_AMOUNT <
	ui_type = "drag";
	ui_min = 1.0; ui_max = 10.0;
    ui_step = 1.0;
    ui_label = "Ray Count";
	ui_tooltip = "Number of rays cast for each pixel.";
    ui_category = "Path Tracing";
> = 4.0;

uniform float RT_RAY_STEPS <
	ui_type = "drag";
	ui_min = 5.0; ui_max = 20.0;
    ui_step = 1.0;
    ui_label = "Ray Step Count";
	ui_tooltip = "Number of steps marched for each ray.";
    ui_category = "Path Tracing";
> = 8.0;

uniform float RT_Z_THICKNESS <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 5.0;
    ui_step = 0.01;
    ui_label = "Object Thickness";
	ui_tooltip = "The shader can't know how thick objects are, since it only\nsees the side the camera faces and has to assume a fixed value.\n\nUse this parameter to remove halos around thin objects.";
    ui_category = "Path Tracing";
> = 0.5;

uniform float RT_FADE_DEPTH_START <
	ui_type = "drag";
	ui_min = 0.00; ui_max = 1.00;
    ui_step = 0.01;
    ui_label = "Fade Out Start";
	ui_tooltip = "Distance where GI starts to fade out.";
    ui_category = "Blending";
> = 0.5;

uniform float RT_FADE_DEPTH_END <
	ui_type = "drag";
	ui_min = 0.00; ui_max = 1.00;
    ui_step = 0.01;
    ui_label = "Fade Out End";
	ui_tooltip = "Distance where GI is completely faded out.";
    ui_category = "Blending";
> = 1.0;

#if INFINITE_BOUNCES != 0
    uniform float RT_IL_BOUNCE_WEIGHT <
        ui_type = "drag";
        ui_min = 0; ui_max = 5.0;
        ui_step = 0.01;
        ui_label = "Next Bounce Weight";
        ui_category = "Blending";
    > = 1.0;
#endif

uniform float RT_AO_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 2.0;
    ui_step = 0.01;
    ui_label = "Ambient Occlusion Intensity";
	ui_tooltip = "Intensity of the darkness added by ambient occlusion.";
    ui_category = "Blending";
> = 1.0;

uniform float RT_IL_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 6.0;
    ui_step = 0.01;
    ui_label = "Indirect Lighting Intensity";
	ui_tooltip = "Intensity of the light absorbed by nearby objects.";
    ui_category = "Blending";
> = 3.0;

/*=============================================================================
	Textures, Samplers, Globals
=============================================================================*/

float2 GetAspectRatio() { return float2(1.0, BUFFER_WIDTH * BUFFER_RCP_HEIGHT); }
float2 GetPixelSize()   { return float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT); }
float2 GetScreenSize()  { return float2(BUFFER_WIDTH, BUFFER_HEIGHT); }
#define ASPECT_RATIO    GetAspectRatio()
#define PIXEL_SIZE      GetPixelSize()
#define SCREEN_SIZE     GetScreenSize()
#define RESHADE_DEPTH_LINEARIZATION_FAR_PLANE 1000.0

uniform int framecount < source = "framecount"; >;

texture BackBufferTex       : COLOR;
texture DepthBufferTex      : DEPTH;
texture ZTex 	            { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F;  MipLevels = 3 + MXAO_MIPLEVEL_AO;};
texture NormalTex 	        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA8; MipLevels = 3 + MXAO_MIPLEVEL_IL;};
texture GITex	            { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITexPrev	        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITexTemp	        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GBufferTexPrev      { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture JitterTex           < source = "LDR_RGB1_18.png"; > { Width = 32; Height = 32; Format = RGBA8; };

sampler sBackBufferTex 	    { Texture = BackBufferTex; 	};
sampler sDepthBufferTex     { Texture = DepthBufferTex; };
sampler2D sZTex	            { Texture = ZTex;	    };
sampler2D sNormalTex	    { Texture = NormalTex;	};
sampler2D sGITex	        { Texture = GITex;	        };
sampler2D sGITexPrev	    { Texture = GITexPrev;	    };
sampler2D sGITexTemp	    { Texture = GITexTemp;	    };
sampler2D sGBufferTexPrev	{ Texture = GBufferTexPrev;	};
sampler	sJitterTex          { Texture = JitterTex; AddressU = WRAP; AddressV = WRAP;};

/*=============================================================================
	Vertex Shader
=============================================================================*/

struct VSOUT
{
	float4                  vpos        : SV_Position;
    float2                  uv          : TEXCOORD0;
    //float2                  uv_scaled   : TEXCOORD1;
    nointerpolation float3  uvtoviewADD : TEXCOORD2;
    nointerpolation float3  uvtoviewMUL : TEXCOORD3;
    nointerpolation float4  viewtouv    : TEXCOORD4;
};

VSOUT VS_RT(in uint id : SV_VertexID)
{
    VSOUT o;

    o.uv.x = (id == 2) ? 2.0 : 0.0;
    o.uv.y = (id == 1) ? 2.0 : 0.0;

    o.vpos = float4(o.uv.xy * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

    o.uvtoviewADD = float3(-1.0,-1.0,1.0);
    o.uvtoviewMUL = float3(2.0,2.0,0.0);

#if 1
    static const float FOV = 75; //vertical FoV
    o.uvtoviewADD = float3(-tan(radians(FOV * 0.5)).xx,1.0) * ASPECT_RATIO.yxx;
   	o.uvtoviewMUL = float3(-2.0 * o.uvtoviewADD.xy,0.0);
#endif

	o.viewtouv.xy = rcp(o.uvtoviewMUL.xy);
    o.viewtouv.zw = -o.uvtoviewADD.xy * o.viewtouv.xy;

    return o;
}

/*=============================================================================
	Functions
=============================================================================*/

struct Ray 
{
    float3 pos;
    float3 dir;
    float2 uv;
    float len;
};

struct MRT
{
    float4 gi   : SV_Target0;
    float4 gbuf : SV_Target1;
};

struct RTConstants
{
    float3 pos;
    float3 normal;
    float3x3 mtbn;
    int nrays;
    int nsteps;
};

float depth_from_uv(float2 uv)
{
    if (RT_FLIP_DEPTH) uv.y = 1.0 - uv.y;

    float depth = tex2Dlod(sDepthBufferTex, float4(uv, 0, 0)).x;

    if (!RT_ILWERT_DEPTH) depth = 1.0 - depth;	//most games already use ilwerted	

    const float N = 1.0;
	depth /= RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - depth * (RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - N);

	return depth;
}

float depth_to_distance(in float depth)
{
	depth = depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE + 1;
	
	return depth;
}

float3 get_position(in VSOUT i, in float2 uv, in float z)
{
    return (uv.xyx * i.uvtoviewMUL + i.uvtoviewADD) * z;
}

float3 get_position_from_uv(in VSOUT i)
{
    return (i.uv.xyx * i.uvtoviewMUL + i.uvtoviewADD) * depth_to_distance(depth_from_uv(i.uv));
}

float3 get_position_from_uv(in VSOUT i, in float2 uv)
{
    return (uv.xyx * i.uvtoviewMUL + i.uvtoviewADD) * depth_to_distance(depth_from_uv(uv));
}

float3 get_position_from_uv(in VSOUT i, in float2 uv, in int mip)
{
    return (uv.xyx * i.uvtoviewMUL + i.uvtoviewADD) * tex2Dlod(sZTex, float4(uv.xyx, mip)).x;
}

float2 get_uv_from_position(in VSOUT i, in float3 pos)
{
	return (pos.xy / pos.z) * i.viewtouv.xy + i.viewtouv.zw;
}

float3 get_normal_from_depth(in VSOUT i)
{
    float3 d = float3(PIXEL_SIZE, 0);
 	float3 pos = get_position_from_uv(i);
	float3 ddx1 = -pos + get_position_from_uv(i, i.uv + d.xz);
	float3 ddx2 = pos - get_position_from_uv(i, i.uv - d.xz);
	float3 ddy1 = -pos + get_position_from_uv(i, i.uv + d.zy);
	float3 ddy2 = pos - get_position_from_uv(i, i.uv - d.zy);

    ddx1 = abs(ddx1.z) > abs(ddx2.z) ? ddx2 : ddx1;
    ddy1 = abs(ddy1.z) > abs(ddy2.z) ? ddy2 : ddy1;

    float3 n = cross(ddy1, ddx1);
    n *= rsqrt(dot(n, n) + 1e-9);
    return n;
}

float3x3 get_tbn(float3 n)
{
    float3 e1 = float3(1, 0, 0);
    float3 e2 = float3(0, 1, 0);
    float3 e3 = float3(0, 0, 1);

    float3 e1proj = e1 - dot(e1, n) * n;
    float3 e2proj = e2 - dot(e2, n) * n;
    float3 e3proj = e3 - dot(e3, n) * n;

    float3 longest = dot(e1proj, e1proj) > dot(e2proj, e2proj) ? e1proj : e2proj;
    longest = dot(longest, longest) > dot(e3proj, e3proj) ? longest : e3proj;

    longest = normalize(longest);

    float3 t = longest;
    float3 b = cross(t, n);

    return float3x3(t, b, n);
}

void unpack_hdr(inout float3 color)
{
    color /= 1.01 - color; //min(min(color.r, color.g), color.b);//max(max(color.r, color.g), color.b);
}

void pack_hdr(inout float3 color)
{
    color /= 1.01 + color; //min(min(color.r, color.g), color.b);//max(max(color.r, color.g), color.b);
}

float compute_temporal_coherence(MRT lwrr, MRT prev)
{
    float4 gbuf_delta = abs(lwrr.gbuf - prev.gbuf);

    float coherence = dot(gbuf_delta.xyz, gbuf_delta.xyz) * 10 + gbuf_delta.w * 5000;
    coherence = exp(-coherence);

    coherence = saturate(1 - coherence);
    return lerp(0.05, 0.9, coherence);
/*
    float coherence = dot(gbuf_delta.xyz, gbuf_delta.xyz) * 10 + gbuf_delta.w * 5000 * saturate(dot(gbuf_delta.xyz, gbuf_delta.xyz) * 100000);
    coherence = exp(-coherence);

    coherence = saturate(1 - coherence);
    return lerp(0.05, 0.9, coherence);*/
}

float3 get_spatiotemporal_jitter(in VSOUT i)
{
    float3 jitter = tex2Dfetch(sJitterTex, int4(i.vpos.xy % tex2Dsize(sJitterTex, 0), 0, 0)).xyz;
    //reduce framecount range to minimize floating point errors
    jitter += (framecount  % 1000) * 3.1;
    return frac(jitter);
}

float4 fetch_gbuffer(in float2 uv)
{
    return float4(tex2D(sNormalTex, uv).xyz * 2 - 1, 
                  tex2D(sZTex, uv).x / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
}

float4 atrous(int iter, sampler gisampler, VSOUT i)
{
    float4 gbuf_center = fetch_gbuffer(i.uv);
    float4 gi_center = tex2D(gisampler, i.uv);

    float z_grad = abs(gbuf_center.z) * rsqrt(dot(gbuf_center.xy, gbuf_center.xy) + 1e-6) + 1e-6;//cot(angle) = cos/sin, cos = z of normal, sin = length of xy of normal

    float4 sum_gi = 0;
    float sum_w = 0.005;

    for(int x = -2; x <= 2; x++)
    for(int y = -2; y <= 2; y++)
    {
        float2 colwuv = i.uv + float2(x,y) * PIXEL_SIZE * exp2(iter);

        float4 gi_tap       = tex2Dlod(gisampler,    float4(colwuv, 0, 0));
        float4 gbuf_tap     = fetch_gbuffer(colwuv);

        float wz = exp(-abs(gbuf_tap.w - gbuf_center.w)/z_grad * 6);
        float wn = pow(saturate(dot(gbuf_tap.xyz, gbuf_center.xyz)), 64);
        float wi = exp(-abs(gi_tap.w - gi_center.w) * 16.0);

        wn = lerp(wn, 1, saturate(1 - wz));

        sum_gi += gi_tap * wz*wn*wi;
        sum_w += wz*wn*wi;
    }

    return sum_gi / sum_w;
}

/*=============================================================================
	Pixel Shaders
=============================================================================*/

void PS_Deferred(in VSOUT i, out float4 normal : SV_Target0, out float depth : SV_Target1)
{	
    normal  = float4(get_normal_from_depth(i) * 0.5 + 0.5, 1); 
    depth   = depth_to_distance(depth_from_uv(i.uv));
}

void PS_RTMain(in VSOUT i, out float4 o : SV_Target0)
{
    RTConstants rtconstants;
    rtconstants.pos     = get_position_from_uv(i, i.uv);
    rtconstants.normal  = tex2D(sNormalTex, i.uv).xyz * 2 - 1;
    rtconstants.mtbn    = get_tbn(rtconstants.normal);
    rtconstants.nrays   = RT_RAY_AMOUNT;
    rtconstants.nsteps  = RT_RAY_STEPS;  

    float3 jitter = get_spatiotemporal_jitter(i);

    float depth = rtconstants.pos.z / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    rtconstants.pos = rtconstants.pos * 0.998 + rtconstants.normal * depth;

    float2 sample_dir;
    sincos(38.39941 * jitter.x, sample_dir.x, sample_dir.y); //2.3999632 * 16 

    MRT lwrr, prev;
    lwrr.gbuf = float4(rtconstants.normal, depth);  
    prev.gi = tex2D(sGITexPrev, i.uv);
    prev.gbuf = tex2D(sGBufferTexPrev, i.uv); 
    float alpha = compute_temporal_coherence(lwrr, prev);

    rtconstants.nrays += 8 * saturate(alpha); //drastically increase quality on areas where temporal filter fails   

    lwrr.gi = 0;

    float ilwthickness = 1.0 / (RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS * RT_Z_THICKNESS);

    [loop]
    for(float r = 0; r < rtconstants.nrays; r++)
    {
        Ray ray; 
        ray.dir.z = (r + jitter.y) / rtconstants.nrays;
        ray.dir.z = sqrt(ray.dir.z); //"cosine" weighting, instead of weighting samples, modulate ray density and use uniform weighting
        ray.dir.xy = sample_dir * sqrt(1 - ray.dir.z * ray.dir.z);
        ray.dir = mul(ray.dir, rtconstants.mtbn);
        ray.len = RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS;
        sample_dir = mul(sample_dir, float2x2(0.76465, -0.64444, 0.64444, 0.76465)); 

        float intersected = 0, mip = 0; int s = 0; bool inside_screen = 1;

        while(s++ < rtconstants.nsteps && inside_screen)
        {
            float lambda = float(s + jitter.z) / rtconstants.nsteps; //normalized position in ray [0, 1]
            lambda *= lambda * rsqrt(lambda); //lambda ^ 1.5 using the fastest instruction sets

            ray.pos = rtconstants.pos + ray.dir * lambda * ray.len;

            ray.uv = get_uv_from_position(i, ray.pos);
            inside_screen = all(saturate(-ray.uv * ray.uv + ray.uv));

            mip = length((ray.uv - i.uv) * ASPECT_RATIO.yx) * 28;
            float3 delta = get_position_from_uv(i, ray.uv, mip + MXAO_MIPLEVEL_AO) - ray.pos;
            
            delta *= ilwthickness;
            [branch]
            if(delta.z < 0 && delta.z > -1)
            {                
                intersected = saturate(1 - dot(delta, delta)) * inside_screen; 
                s = rtconstants.nsteps;
            }           
        }

        lwrr.gi.w += intersected;    

        [branch]
        if(RT_IL_AMOUNT > 0 && intersected > 0.05)
        {
            float3 albedo 			= tex2Dlod(sBackBufferTex, 	float4(ray.uv, 0, mip)).rgb; unpack_hdr(albedo);
            float3 intersect_normal = tex2Dlod(sNormalTex,      float4(ray.uv, 0, mip)).xyz * 2.0 - 1.0;

 #if INFINITE_BOUNCES != 0
            float3 nextbounce 		= tex2Dlod(sGITexPrev, float4(ray.uv, 0, 0)).rgb; unpack_hdr(nextbounce);            
            albedo += nextbounce * RT_IL_BOUNCE_WEIGHT;
#endif
            lwrr.gi.rgb += albedo * intersected * saturate(dot(-intersect_normal, ray.dir));
        }
    }
    lwrr.gi /= rtconstants.nrays; 
    pack_hdr(lwrr.gi.rgb);

    o = lerp(prev.gi, lwrr.gi, alpha);
}

void PS_Copy(in VSOUT i, out MRT o)
{	
    o.gi    = tex2D(sGITex, i.uv);
    o.gbuf  = fetch_gbuffer(i.uv);
}

void PS_Filter0(in VSOUT i, out float4 o : SV_Target0)
{
    o = atrous(0, sGITexPrev, i);
}
void PS_Filter1(in VSOUT i, out float4 o : SV_Target0)
{
    o = atrous(1, sGITexTemp, i);
}
void PS_Filter2(in VSOUT i, out float4 o : SV_Target0)
{
    o = atrous(2, sGITex, i);
}

//need this as backbuffer is not guaranteed to have RGBA8
void PS_Output(in VSOUT i, out float4 o : SV_Target0)
{
#if SPATIAL_FILTER != 0
    float4 gi = tex2D(sGITexTemp, i.uv);
#else
    float4 gi = tex2D(sGITex, i.uv);
#endif
    float3 color = tex2D(sBackBufferTex, i.uv).rgb;
	
#if DEBUG_TESTING != 0
	if (RT_FILTER_ACTIVE)
	{
#endif

	unpack_hdr(color);
	unpack_hdr(gi.rgb);

	gi *= smoothstep(RT_FADE_DEPTH_END, RT_FADE_DEPTH_START, depth_from_uv(i.uv));

	gi.w = RT_AO_AMOUNT > 1 ? pow(1 - gi.w, RT_AO_AMOUNT) : 1 - gi.w * RT_AO_AMOUNT;
	gi.rgb *= RT_IL_AMOUNT * RT_IL_AMOUNT;

	color = color * gi.w * (1 + gi.rgb);  

#if DEBUG_TESTING != 0
	if(RT_DEBUG_VIEW)
		color.rgb = gi.w * (1 + gi.rgb);
#endif

	pack_hdr(color.rgb);
	
#if DEBUG_TESTING != 0
	}
#endif

    o = float4(color, 1);
}

/*=============================================================================
	Techniques
=============================================================================*/

technique RT
{
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Deferred;
        RenderTarget0 = NormalTex;
        RenderTarget1 = ZTex;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_RTMain;
        RenderTarget0 = GITex;
	}  
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Copy;
        RenderTarget0 = GITexPrev;
        RenderTarget1 = GBufferTexPrev;
	}
#if SPATIAL_FILTER != 0
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Filter0;
        RenderTarget = GITexTemp;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Filter1;
        RenderTarget = GITex;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Filter2;
        RenderTarget = GITexTemp;
	}
#endif
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Output;
	}
}