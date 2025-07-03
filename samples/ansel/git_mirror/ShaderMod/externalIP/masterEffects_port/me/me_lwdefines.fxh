struct VSOut
{
    float4 position : SV_Position;
    float2 texcoord: TexCoord;
};

cbuffer globalParams
{
	float2 screenSize;
	float elapsedTime;
	int captureState;
}

SamplerState colorLinearSampler;
SamplerState colorLinearSamplerWrap;

float4 GaussBlur22_2D(float2 coord, Texture2D tex, SamplerState sampState, float mult, float lodlevel, bool isBlurVert) //texcoord, texture, sampler, blurmult in pixels, tex2dlod level, axis (0=horiz, 1=vert)
{
	float4 sum = 0;
	float2 axis = (isBlurVert) ? float2(0, 1) : float2(1, 0);
	float  weight[11] = {0.082607, 0.080977, 0.076276, 0.069041, 0.060049, 0.050187, 0.040306, 0.031105, 0.023066, 0.016436, 0.011254};

	for(int i=-10; i < 11; i++)
	{
		float lwrrweight = weight[abs(i)];	
		//PORTED: sum	+= tex2Dlod(tex, float4(coord.xy + axis.xy * (float)i * PixelSize * mult,0,lodlevel)) * lwrrweight;
		sum	+= tex.SampleLevel(sampState, coord.xy + axis.xy * (float)i * PixelSize * mult, lodlevel) * lwrrweight;
	}

	return sum;
}

float smootherstep(float edge0, float edge1, float x)
{
   	x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
   	return x*x*x*(x*(x*6 - 15) + 10);
}

float3 Hue(in float3 RGB)
{
   	// Based on work by Sam Hocevar and Emil Persson
   	float Epsilon = 1e-10;
   	float4 P = (RGB.g < RGB.b) ? float4(RGB.bg, -1.0, 2.0/3.0) : float4(RGB.gb, 0.0, -1.0/3.0);
   	float4 Q = (RGB.r < P.x) ? float4(P.xyw, RGB.r) : float4(RGB.r, P.yzx);
   	float C = Q.x - min(Q.w, Q.y);
   	float H = abs((Q.w - Q.y) / (6 * C + Epsilon) + Q.z);
   	return float3(H, C, Q.x);
}


