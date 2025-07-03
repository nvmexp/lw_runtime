////////////////////////////////////////////////////////////////////////////////
// Filename: shader001.fx
////////////////////////////////////////////////////////////////////////////////


/////////////
// GLOBALS //
/////////////
matrix worldMatrix;
matrix viewMatrix;
matrix projectionMatrix;
float4 vMeshColor;

//Texture2D txArray[3];
Texture2D txArray0,txArray1,txArray2;

SamplerState samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

//////////////
// TYPEDEFS //
//////////////
struct VertexInputType
{
    float4 position : POSITION;
    float2 texcoord : TEXCOORD;
    uint texarrayindex : TEXARRAYINDEX;
};

struct PixelInputType
{
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
    uint texarrayindex : TEXARRAYINDEX;
};


////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
PixelInputType LineVertexShader(VertexInputType input)
{
    PixelInputType output =(PixelInputType)0;
    
    output.position = mul(input.position, worldMatrix);
    output.position = mul(output.position, viewMatrix);
    output.position = mul(output.position, projectionMatrix);
    
    output.texcoord = input.texcoord;
    
    output.texarrayindex = input.texarrayindex;
    
    return output;
}


////////////////////////////////////////////////////////////////////////////////
// Pixel Shader
////////////////////////////////////////////////////////////////////////////////
float4 LinePixelShader(PixelInputType input) : SV_Target
{
    if (input.texarrayindex == 0)
    {
        return txArray0.Sample( samLinear, input.texcoord ) * vMeshColor;
    }
    else if (input.texarrayindex == 1)
    { 
        return txArray1.Sample( samLinear, input.texcoord ) * vMeshColor;
    }
    else if (input.texarrayindex == 2)
    {
        return txArray2.Sample( samLinear, input.texcoord ) * vMeshColor;
    }
    
    return (txArray0.Sample( samLinear, input.texcoord )) * vMeshColor;	
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
    }
}

