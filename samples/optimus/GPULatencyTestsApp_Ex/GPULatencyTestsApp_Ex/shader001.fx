////////////////////////////////////////////////////////////////////////////////
// Filename: shader001.fx
////////////////////////////////////////////////////////////////////////////////

/////////////
// GLOBALS //
/////////////
matrix worldMatrix;
matrix viewMatrix;
matrix projectionMatrix;

Texture3D txWaste;

SamplerState samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

/////////////////////
// BLENDING STATES //
/////////////////////
BlendState AlphaBlendingOff
{
    BlendEnable[0] = FALSE;
};


//////////////
// TYPEDEFS //
//////////////
struct VertexInputType
{
    float4 position : POSITION;
    float4 color : COLOR;
};

struct PixelInputType
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
};


////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
PixelInputType LineVertexShader(VertexInputType input)
{
    PixelInputType output;
    
    input.position.w = 1.0f;

    output.position = mul(input.position, worldMatrix);
    output.position = mul(output.position, viewMatrix);
    output.position = mul(output.position, projectionMatrix);
    
    output.color = input.color;
    
    return output;
}


////////////////////////////////////////////////////////////////////////////////
// Pixel Shader
////////////////////////////////////////////////////////////////////////////////
float4 LinePixelShader(PixelInputType input) : SV_Target
{
    float3 JunkTexCoord;

	JunkTexCoord.x = input.position.x / 512.0f;
	JunkTexCoord.y = input.position.y / 512.0f;
	JunkTexCoord.z = (input.position.x * input.position.y) / 512.0f;

    return input.color + txWaste.Sample( samLinear, JunkTexCoord );
}


////////////////////////////////////////////////////////////////////////////////
// Technique
////////////////////////////////////////////////////////////////////////////////
technique10 LineTechnique
{
    pass pass0
    {
        SetVertexShader(CompileShader(vs_4_0, LineVertexShader()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, LinePixelShader()));
        SetBlendState(AlphaBlendingOff, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xFFFFFFFF);
    }
}

