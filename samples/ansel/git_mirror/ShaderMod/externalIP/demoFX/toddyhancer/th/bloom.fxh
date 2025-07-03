//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Bloom 													     	     //
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

float4 GaussBlur22_2D(float2 coord, Texture2D tex, SamplerState sampState, float mult, float lodlevel, bool isBlurVert) //texcoord, texture, sampler, blurmult in pixels, tex2dlod level, axis (0=horiz, 1=vert)
{
	float4 sum = 0;
	float2 axis = (isBlurVert) ? float2(0, 1) : float2(1, 0);
	float  weight[11] = {0.082607, 0.080977, 0.076276, 0.069041, 0.060049, 0.050187, 0.040306, 0.031105, 0.023066, 0.016436, 0.011254};

	for(int i=-10; i < 11; i++)
	{
		float lwrrweight = weight[abs(i)];	
		sum	+= tex.SampleLevel(sampState, coord.xy + axis.xy * (float)i * PixelSize * mult, lodlevel) * lwrrweight;
	}

	return sum;
}

float4 PS_ME_BloomPrePass(VSOut IN) : SV_Target
{
	
	float4 bloom=0.0;
	float2 bloomuv;

	float2 offset[4]=
	{
		float2(1.0, 1.0),
		float2(1.0, 1.0),
		float2(-1.0, 1.0),
		float2(-1.0, -1.0)
	};

	for (int i=0; i<4; i++)
	{
		bloomuv.xy=offset[i]*PixelSize.xy*2;
		bloomuv.xy=IN.texcoord.xy + bloomuv.xy;
		float4 tempbloom = inputColor.Sample(colorLinearSampler, bloomuv.xy);
		tempbloom.w = max(0,dot(tempbloom.xyz,0.333)-fAnamFlareThreshold);
		tempbloom.xyz = max(0, tempbloom.xyz-fBloomThreshold); 

		bloom+=tempbloom;
	}

	bloom *= 0.25;
	return bloom;
}

float4 PS_ME_BloomPass1(VSOut IN) : SV_Target
{

	float4 bloom=0.0;
	float2 bloomuv;

	float2 offset[8]=
	{
		float2(1.0, 1.0),
		float2(0.0, -1.0),
		float2(-1.0, 1.0),
		float2(-1.0, -1.0),
		float2(0.0, 1.0),
		float2(0.0, -1.0),
		float2(1.0, 0.0),
		float2(-1.0, 0.0)
	};

	for (int i=0; i<8; i++)
	{
		bloomuv.xy=offset[i]*PixelSize.xy*4;
		bloomuv.xy=IN.texcoord.xy + bloomuv.xy;
		float4 tempbloom=SamplerBloom1.Sample(colorLinearSampler, bloomuv.xy);
		bloom+=tempbloom;
	}

	bloom *= 0.125;
	return bloom;
}

float4 PS_ME_BloomPass2(VSOut IN) : SV_Target
{

	float4 bloom=0.0;
	float2 bloomuv;

	float2 offset[8]=
	{
		float2(0.707, 0.707),
		float2(0.707, -0.707),
		float2(-0.707, 0.707),
		float2(-0.707, -0.707),
		float2(0.0, 1.0),
		float2(0.0, -1.0),
		float2(1.0, 0.0),
		float2(-1.0, 0.0)
	};

	for (int i=0; i<8; i++)
	{
		bloomuv.xy=offset[i]*PixelSize.xy*8;
		bloomuv.xy=IN.texcoord.xy + bloomuv.xy;
		//PORTED: float4 tempbloom=tex2Dlod(SamplerBloom2, float4(bloomuv.xy, 0, 0));
		float4 tempbloom=SamplerBloom2.Sample(colorLinearSampler, bloomuv.xy);
		bloom+=tempbloom;
	}

	bloom *= 0.5; //to brighten up the sample, it will lose brightness in H/V gaussian blur 
	return bloom;
}


float4 PS_ME_BloomPass3(VSOut IN) : SV_Target
{
	float4 bloom;
	// PORTED
	bloom = GaussBlur22_2D(IN.texcoord.xy, SamplerBloom3, colorLinearSampler, 16, 0, 0);
	bloom.a *= fAnamFlareAmount;
	bloom.xyz *= fBloomAmount;
	return bloom;
}

float4 PS_ME_BloomPass4(VSOut IN) : SV_Target
{
	float4 bloom;
	// PORTED
	bloom.xyz = GaussBlur22_2D(IN.texcoord.xy, SamplerBloom4, colorLinearSampler, 16, 0, 1).xyz*2.5;	
	bloom.w   = GaussBlur22_2D(IN.texcoord.xy, SamplerBloom4, colorLinearSampler, 32*fAnamFlareWideness, 0, 0).w*2.5; //to have anamflare texture (bloom.w) avoid vertical blur
	return bloom;
}