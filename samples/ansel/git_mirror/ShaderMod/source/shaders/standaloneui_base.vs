struct VSIn
{
   float4 position : POSITION;
   float2 texcoord : TEXCOORD;
};

cbuffer ControlBuf : register(b0)
{
	float4 g_Color: packoffset(c0);
	float g_OffsetX: packoffset(c1);
	float g_OffsetY: packoffset(c1.y);
	float g_ScaleX: packoffset(c1.z);
	float g_ScaleY: packoffset(c1.w);
}

struct VSOut
{
   float4 position : SV_POSITION;
   float2 texcoord : TEXCOORD;
   float4 color : COLOR;
};

VSOut main( VSIn vertex )
{
	VSOut output;

	output.position = float4(g_ScaleX, g_ScaleY, 1.0, 1.0) * vertex.position + float4(g_OffsetX, g_OffsetY, 0.0, 0.0);
	output.texcoord = vertex.texcoord;
	output.color = g_Color;

	return output;
}