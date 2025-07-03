Texture2D g_txDiffuse : register(t0);
Texture2D g_txBump: register(t1);
Texture2D g_txElw: register(t2);
Texture2D g_txLobby: register(t0);

cbuffer cbEveryFrame : register(b0)
{
    float4x4 g_mWorldViewProjection;    // World * View * Projection matrix
    float4x4 g_mIlwWorldViewProjection;
    float4x4 g_mWorld;
	float4x4 g_mWorldView;
    float3 g_vEyePt;
};

//-----------------------------------------------------------------------------
// Vertex shader structures
//-----------------------------------------------------------------------------

struct VS_INPUT_GEO
{
    float3 Pos           : POSITION;
    float2 Tex1          : TEXCOORD1;
};

struct VS_OUTPUT_GEO
{
    float4 Pos        : SV_Position;
    float2 Tex0        : TEXCOORD0;
    float2 Tex1        : TEXCOORD1;
};

struct VS_INPUT_ELW
{
   float4 Pos        : POSITION;
   float2 Tex        : TEXCOORD0;
};

struct VS_OUTPUT_ELW
{
    float4 Pos        : SV_Position;
    float2 Tex        : TEXCOORD0;
};

SamplerState g_samStage0 : register(s0);
SamplerState g_samStage1 : register(s1);

VS_OUTPUT_GEO GeometryVS( VS_INPUT_GEO input )
{
    VS_OUTPUT_GEO output;

    float4 wPos = mul( float4(input.Pos,1),g_mWorld );

    output.Pos = mul( float4(input.Pos,1), g_mWorldViewProjection );

	float3 transformedNormal = mul( float4(normalize(input.Pos), 1.0f), g_mWorldView ).xyz;
	float scale = 1.37f / length(transformedNormal);

    output.Tex1 = input.Tex1;

    output.Tex0.x = 0.5f + scale * transformedNormal.x;
	output.Tex0.y = 0.5f - scale * transformedNormal.y;

    return output;
}

float4 GeometryPS( VS_OUTPUT_GEO input ) : SV_Target
{
    float4 Diffuse    = g_txDiffuse.Sample(g_samStage0, input.Tex1);
    float2 Bump       = g_txBump.Sample(g_samStage1, input.Tex1).xy;
    float2 BumpOffset;

    BumpOffset.x = Bump.x * 0.5f;
    BumpOffset.y = Bump.y * 0.5f;

    float2 ElwTexCrd = input.Tex0 + BumpOffset;
    float4 Elw = g_txElw.Sample(g_samStage1, ElwTexCrd);

    return Diffuse + 0.25 * Elw;
}

VS_OUTPUT_ELW ElwVS( VS_INPUT_ELW input )
{
    VS_OUTPUT_ELW output;

    output.Pos = input.Pos;
    output.Tex = input.Tex;

    return output;
}

float4 ElwPS( VS_OUTPUT_ELW input ) : SV_Target
{
    float4 color = g_txLobby.Sample( g_samStage0, input.Tex );
    return color;
}


//-----------------------------------------------------------------------------
// Compute shader structures
//-----------------------------------------------------------------------------

cbuffer cbCompute : register(b0)
{
    float4x4 g_mBackgroundRotation;
    float    g_BackgroundZoom;
};

RWTexture2D<float4> output : register (u0);

[numthreads(4, 64, 1)]
void BackgroundCS( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    float4 outputPos = float4(DTid.x,DTid.y,0,1);
    float4 worldPos = mul(outputPos, g_mBackgroundRotation);
    worldPos /= 30;
    float4 worldFrac = frac(worldPos);
    float4 worldFloor = floor(worldPos);
    if ((worldFloor.x + worldFloor.y) % 2)
    {
        output[DTid.xy] = float4(worldFrac.x, worldFrac.y, 0, 1);
    }
    else
    {
        output[DTid.xy] = float4(1,1,1,1);
    }
}
