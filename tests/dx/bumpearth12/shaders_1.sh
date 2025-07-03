SamplerState g_samStage0 : register(s0, space1);
SamplerState g_samStage1 : register(s1, space1);

Texture2D g_tx0 : register(t10, space2);
Texture2D g_tx1[2]: register(t10, space3);

struct Transforms
{
    float4x4 worldViewProjection;    // World * View * Projection matrix
    float4x4 ilwWorldViewProjection;
    float4x4 world;
    float4x4 worldView;
    float3 eyePt;
};
ConstantBuffer<Transforms> cb0 : register(b2, space4);

struct Constants
{
    float  scaleFactor;
    float  half;
    float  quarter;
    float  zero;
};
//ConstantBuffer<Constants> cb1[2] : register(b2, space5);
ConstantBuffer<Constants> cb1_0 : register(b2, space5);
ConstantBuffer<Constants> cb1_1 : register(b3, space5);


RWByteAddressBuffer uav0   : register(u1, space6);
//RWBuffer<uint> uav1[2] : register(u1, space7);
RWByteAddressBuffer uav1_0 : register(u1, space7);
RWByteAddressBuffer uav1_1 : register(u2, space7);

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
    float4 Pos         : SV_Position;
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

VS_OUTPUT_GEO GeometryVS1( VS_INPUT_GEO input )
{
    VS_OUTPUT_GEO output;

    float4 wPos = mul( float4(input.Pos,1),cb0.world );

    output.Pos = mul( float4(input.Pos,1), cb0.worldViewProjection );

    uint x = (uint)(wPos.x * 1000);
    uint y = (uint)(wPos.x * 1000) + 1;
    uint cbIdx0 = (x % 2);
    uint cbIdx1 = (y % 2);

	float3 transformedNormal = mul( float4(normalize(input.Pos), 1.0f), cb0.worldView ).xyz;
	float scale = cb1_0.scaleFactor / length(transformedNormal);

    output.Tex1 = input.Tex1;

    output.Tex0.x = cb1_0.half + scale * transformedNormal.x;
	output.Tex0.y = cb1_1.half - scale * transformedNormal.y;

    return output;
}

float4 GeometryPS1( VS_OUTPUT_GEO input ) : SV_Target
{
    uint x = (uint)(input.Pos.x * 1000);
    uint y = (uint)(input.Pos.x * 1000) + 1;
    uint cbIdx0 = (x % 2);
    uint cbIdx1 = (y % 2);

    float4 Diffuse    = g_tx0.Sample(g_samStage0, input.Tex1);
    float2 Bump;
    if (cbIdx0 == 0)
        Bump = g_tx1[cbIdx0].Sample(g_samStage0, input.Tex1).xy;
    else
        Bump = g_tx1[cbIdx0].Sample(g_samStage1, input.Tex1).xy;
    float2 BumpOffset;

    BumpOffset.x = Bump.x * cb1_0.half;
    BumpOffset.y = Bump.y * cb1_1.half;

    float2 ElwTexCrd = input.Tex0 + BumpOffset;
    float4 Elw;
    if (cbIdx0 == 0)
        Elw = g_tx1[cbIdx1].Sample(g_samStage0, ElwTexCrd);
    else
        Elw = g_tx1[cbIdx1].Sample(g_samStage1, ElwTexCrd);

    uint Sampled0 = uav0.Load(cbIdx0<<2);
    uint Sampled1 = uav1_0.Load(cbIdx1<<2);
    uint Sampled2 = uav1_1.Load(cbIdx1<<2);
    Diffuse.x = saturate(Diffuse.x + (Sampled0 / 255.0f));
    Diffuse.y = saturate(Diffuse.y + (Sampled1 / 255.0f));
    Diffuse.z = saturate(Diffuse.z + (Sampled2 / 255.0f));

    return Diffuse + cb1_1.quarter * Elw;
}

VS_OUTPUT_ELW ElwVS1( VS_INPUT_ELW input )
{
    VS_OUTPUT_ELW output;

    output.Pos = input.Pos;
    output.Tex = input.Tex;

    return output;
}

float4 ElwPS1( VS_OUTPUT_ELW input ) : SV_Target
{
    float4 color = g_tx1[1].Sample( g_samStage0, input.Tex );
    return color;
}
