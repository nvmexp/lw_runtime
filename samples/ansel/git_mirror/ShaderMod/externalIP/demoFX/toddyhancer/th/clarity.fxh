//High Pass Sharpening & Contrast Enhancement
//Version 1.2
/* This shader uses the gaussian blur passes from Boulotaur2024's gaussian blur/ bloom, unsharpmask shader which are based
   on the implementation in the article "Efficient Gaussian blur with linear sampling"
   http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/ .
   The blend modes are based on algorithms found at http://www.dunnbypaul.net/blends/ , 
   http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html ,
   http://www.simplefilter.de/en/basics/mixmods.html and http://en.wikipedia.org/wiki/Blend_modes . 
   For more info go to http://reshade.me/forum/shader-presentation/529-high-pass-sharpening */ 
#define CoefLuma_HP float3(0.2126, 0.7152, 0.0722)



#if ClarityTextureFormat == 1
#define CETexFormat R16F
#elif ClarityTextureFormat == 2 
#define CETexFormat R32F
#else
#define CETexFormat R8
#endif

#if ClarityTexScale == 1 
#define CEscale 0.5 
#elif ClarityTexScale == 2 
#define CEscale 0.25 
#else 
#define CEscale 1
#endif

#define cePX_SIZE (RFX_PixelSize*ClarityOffset)

//texture ceBlurTex2Dping{ Width = BUFFER_WIDTH*CEscale; Height = BUFFER_HEIGHT*CEscale; Format = CETexFormat; };
//sampler2D ceBlurSamplerPing { Texture = ceBlurTex2Dping; MinFilter = Linear; MagFilter = Linear; MipFilter = Linear; AddressU = Clamp; SRGBTexture = FALSE;};

float3 argb2hsv(float3 rgb) {
    float milwalue = min(min(rgb.r, rgb.g), rgb.b);
    float maxValue = max(max(rgb.r, rgb.g), rgb.b);
    float d = maxValue - milwalue;

    float3 hsv = 0.0;
    hsv.b = maxValue;
    if (d != 0) {
        hsv.g = d / maxValue;

        float3 delrgb = (((maxValue.xxx - rgb) / 6.0) + d / 2.0) / d;
        if      (maxValue == rgb.r) { hsv.r = delrgb.b - delrgb.g; }
        else if (maxValue == rgb.g) { hsv.r = 1.0 / 3.0 + delrgb.r - delrgb.b; }
        else if (maxValue == rgb.b) { hsv.r = 2.0 / 3.0 + delrgb.g - delrgb.r; }

        if (hsv.r < 0.0) { hsv.r += 1.0; }
        if (hsv.r > 1.0) { hsv.r -= 1.0; }
    }
    return saturate(hsv);
}

float3 ahsv2rgb(float3 hsv) {
    const float h = hsv.r, s = hsv.g, v = hsv.b;

    float3 rgb = v;
    if (hsv.g != 0.0) {
        float h_i = floor(6 * h);
        float f = 6 * h - h_i;

        float p = v * (1.0 - s);
        float q = v * (1.0 - f * s);
        float t = v * (1.0 - (1.0 - f) * s);

        if      (h_i == 0) { rgb = float3(v, t, p); }
        else if (h_i == 1) { rgb = float3(q, v, p); }
        else if (h_i == 2) { rgb = float3(p, v, t); }
        else if (h_i == 3) { rgb = float3(p, q, v); }
        else if (h_i == 4) { rgb = float3(t, p, v); }
        else               { rgb = float3(v, p, q); }
    }
    return saturate(rgb);
}

float4 PS_ClarityFinal(VSOut IN) : SV_Target
{
	if (g_chkClarity)
	{////////////////////////////////////////////////////

#if ClarityRadius == 0
	float sampleOffsets[5] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130 };
	float sampleWeights[5] = { 0.16818994, 0.27276957, 0.11690125, 0.024067905, 0.0021112196 };
	#define SamplesG 5
#elif ClarityRadius == 1
	float sampleOffsets[9] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130, 9.0869565, 11, 12.9130435, 14.826087 };
	float sampleWeights[9] = { 0.1097184698841, 0.11815835218275, 0.10971846988355, 0.087774775907115, 0.0603451584361505, 0.035497152021265, 0.017748576010605, 0.0074730846360555, 0.0026155796226175 };
	#define SamplesG 9
#elif ClarityRadius == 2
	float sampleOffsets[14] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130, 9.0869565, 11, 12.9130435, 14.826087, 16.7391305, 18.652174, 20.5652175, 22.478261, 24.3913045 };
	float sampleWeights[14] = { 0.07217787046621, 0.0739383063308, 0.07217787046621, 0.0671422050846, 0.0595124090525, 0.050254923194, 0.0404224382261, 0.030961867593, 0.0225763617753, 0.015665230617, 0.01033905220881, 0.006487248442, 0.0038673981121, 0.0021890932698 };
	#define SamplesG 14
#elif ClarityRadius == 3
	float sampleOffsets[49] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130, 9.0869565, 11, 12.9130435, 14.826087, 16.7391305, 18.652174, 20.5652175, 22.478261, 24.3913045, 26.304348, 28.2173915, 30.130435, 32.0434785,  33.956522, 35.8695655, 37.782609, 39.6956525, 41.608696, 43.5217395, 45.434783, 47.3478265, 49.26087, 51.1739135, 53.086957, 55.0000005, 56.913044, 58.8260875, 60.739131, 62.6521745, 62.6521745, 66.4782615, 68.391305, 70.3043485, 72.217392, 74.1304355, 76.043479, 77.9565225, 79.869566, 81.7826095, 83.695653, 85.6086965, 87.52174, 89.4347835, 91.347827 };
	float sampleWeights[27] = { 0.038588721157425, 0.04019658453898,  0.0393926528482025, 0.038588721157425, 0.0370754379748375, 0.03556215479225, 0.033510492016125, 0.03145882924, 0.02908457797625, 0.0267103267125, 0.02423714831875, 0.021763969925, 0.01938971865625, 0.0170154673875, 0.01488853396875, 0.01276160055, 0.0109704987125, 0.009179396875, 0.0077550077065, 0.006330618538, 0.00525763234525, 0.0041846461525, 0.0034174610244375, 0.002650275896375, 0.0021289101456875, 0.001607544395, 0.001270478635 };
	#define SamplesG 27
#elif ClarityRadius >= 4
	float sampleOffsets[49] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130, 9.0869565, 11, 12.9130435, 14.826087, 16.7391305, 18.652174, 20.5652175, 22.478261, 24.3913045, 26.304348, 28.2173915, 30.130435, 32.0434785,  33.956522, 35.8695655, 37.782609, 39.6956525, 41.608696, 43.5217395, 45.434783, 47.3478265, 49.26087, 51.1739135, 53.086957, 55.0000005, 56.913044, 58.8260875, 60.739131, 62.6521745, 62.6521745, 66.4782615, 68.391305, 70.3043485, 72.217392, 74.1304355, 76.043479, 77.9565225, 79.869566, 81.7826095, 83.695653, 85.6086965, 87.52174, 89.4347835, 91.347827 };
	float sampleWeights[53] = { 0.03979461869359125*0.533, 0.04019658453898*0.533,  0.03979461869359125*0.533, 0.0393926528482025*0.533, 0.03899068700281375*0.533, 0.038588721157425*0.533, 0.03783207956613125*0.533, 0.0370754379748375*0.533, 0.03631879638354375*0.533, 0.03556215479225*0.533, 0.0345363234041875*0.533, 0.033510492016125*0.533, 0.0324846606280625*0.533, 0.03145882924*0.533, 0.030271703608125*0.533, 0.02908457797625*0.533, 0.027897452344375*0.533, 0.0267103267125*0.533, 0.025473737515625*0.533, 0.02423714831875*0.533, 0.023000559121875*0.533, 0.021763969925*0.533, 0.020576844290625*0.533, 0.01938971865625*0.533, 0.018202593021875*0.533, 0.0170154673875*0.533, 0.015952000678125*0.533, 0.01488853396875*0.533, 0.013825067259375*0.533, 0.01276160055*0.533, 0.01186604963125*0.533, 0.0109704987125*0.533, 0.01007494779375*0.533, 0.009179396875*0.533, 0.00846720229075*0.533, 0.0077550077065*0.533, 0.00704281312225*0.533, 0.006330618538*0.533, 0.005794125441625*0.533, 0.00525763234525*0.533, 0.004721139248875*0.533, 0.0041846461525*0.533, 0.00380105358846875*0.533, 0.0034174610244375*0.533, 0.00303386846040625*0.533, 0.002650275896375*0.533, 0.00238959302103125*0.533, 0.0021289101456875*0.533, 0.00186822727034375*0.533, 0.001607544395*0.533, 0.001439011515*0.533, 0.001270478635 };
	#define SamplesG 49
#endif

		float color = ceBlurSamplerPing.Sample(colorLinearSampler, IN.texcoord.xy).r * sampleWeights[0];
		
		[loop]
		for(int j = 1; j < SamplesG; ++j) {
			color += ceBlurSamplerPing.Sample(colorLinearSampler, IN.texcoord.xy + float2(0.0, sampleOffsets[j] * cePX_SIZE.y)).r * sampleWeights[j];
			color += ceBlurSamplerPing.Sample(colorLinearSampler, IN.texcoord.xy - float2(0.0, sampleOffsets[j] * cePX_SIZE.y)).r * sampleWeights[j];
		}
	
	//color = saturate(color);
	
	float4 origSample = inputColor.Sample(colorLinearSampler, IN.texcoord.xy);
	float3 orig = origSample.rgb; //Original Image
	
#if LightnessTest == 1
	float luma = max(orig.r,max(orig.g,orig.b));
#elif LightnessTest == 2
	float luma = min(orig.r,min(orig.g,orig.b));
#elif LightnessTest == 3 
	float luma = (max(orig.r,max(orig.g,orig.b))+min(orig.r,min(orig.g,orig.b)))*0.5;
#elif LightnessTest == 4
	orig = argb2hsv(orig);
	float luma = orig.b;
#else
	float luma = dot(orig.rgb,CoefLuma_HP);
#endif
	
#if LightnessTest == 4
	float2 chroma = orig.rg; 
#else
	float3 chroma = orig.rgb - luma;
#endif
			
	float sharp = 1-color;
	sharp = (luma+sharp)*0.5;
	
		float sharpMin = min(0.5,(DarkIntensity*(sharp - 0.5)*(sharp+0.5))+0.5);
		float sharpMax = max(0.5,(LightIntensity*(sharp - 0.5)*(sharp+0.5))+0.5);
		sharp = lerp(sharpMin,sharpMax,step(0.5,sharp));
		
		#if ViewMask == 1
				orig.rgb = sharp;
				luma = sharp;
			#elif ClarityBlendMode == 3
				//Multiply
				sharp = 2 * luma * sharp;
			#elif ClarityBlendMode == 6
				//soft light #2
				sharp = lerp(luma*(sharp+0.5),1-(1-luma)*(1-(sharp-0.5)),step(0.5,sharp));
			#elif ClarityBlendMode == 2
				//overlay
				sharp = lerp(2*luma*sharp, 1.0 - 2*(1.0-luma)*(1.0-sharp), step(0.49,luma));
			#elif ClarityBlendMode == 4
				//Hardlight
				//sharp = luma+2*sharp-1;
				sharp = lerp(2*luma*sharp, 1.0 - 2*(1.0-luma)*(1.0-sharp), step(0.49,sharp));
			#elif ClarityBlendMode == 1 
				//softlight
				sharp = lerp(2*luma*sharp + luma*luma*(1.0-2*sharp), 2*luma*(1.0-sharp)+pow(luma,0.5)*(2*sharp-1.0), step(0.49,sharp));
				//sharp = lerp(2*luma*sharp + luma*luma*(1.0-2*sharp), 2*luma*(1.0-sharp)+pow(luma,0.5)*(2*sharp-1.0), smoothstep(0.48,0.49,sharp));
			#elif ClarityBlendMode == 5
				//vivid light
				sharp = lerp(2*luma*sharp, luma/(2*(1-sharp)), step(0.5,sharp));
			#elif ClarityBlendMode == 7
				//soft light #3
				sharp = lerp((2*sharp-1)*(luma-pow(luma,2))+luma, ((2*sharp-1)*(pow(luma,0.5)-luma))+luma, step(0.49,sharp));
			#endif
					
					#if BlendIfDark > 0 || BlendIfLight < 255 || ViewBlendIfMask == 1
						#define BlendIfD (BlendIfDark.0/255.0)+0.0001
						#define BlendIfL (BlendIfLight.0/255.0)-0.0001
						float mix = luma;
					#if ViewBlendIfMask == 1 
						float3 red = lerp(float3(0.0,0.0,1.0),float3(1.0,0.0,0.0),smoothstep(0.0,BlendIfD,mix));
						float3 blue = lerp(red,float3(0.0,0.0,1.0),smoothstep(BlendIfL,1.0,mix));
						orig = blue;
					#else
						sharp = lerp(luma,sharp,smoothstep(0.0,BlendIfD,mix));
						sharp = lerp(sharp,luma,smoothstep(BlendIfL,1.0,mix));
					#endif
					#endif
		
			#if ViewMask == 1 || ViewBlendIfMask == 1
				
			#else 
				#if LightnessTest == 4
					orig.b = lerp(luma, sharp, ClarityStrength);
					orig.rgb = float3(chroma,orig.b);
					orig = ahsv2rgb(orig);
				#else
					orig.rgb = lerp(luma, sharp, ClarityStrength);
					orig.rgb += chroma;
				#endif
				
			#endif 

	return float4(saturate(orig), origSample.a);

	}////////////////////////////////////////////////////
	
	return inputColor.Sample(colorLinearSampler, IN.texcoord.xy);
}	

float4 PS_ClarityBlurX(VSOut IN) : SV_Target
{

#if ClarityRadius == 0
	float sampleOffsets[5] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130 };
	float sampleWeights[5] = { 0.16818994, 0.27276957, 0.11690125, 0.024067905, 0.0021112196 };
	#define SamplesG 5
#elif ClarityRadius == 1
	float sampleOffsets[9] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130, 9.0869565, 11, 12.9130435, 14.826087 };
	float sampleWeights[9] = { 0.1097184698841, 0.11815835218275, 0.10971846988355, 0.087774775907115, 0.0603451584361505, 0.035497152021265, 0.017748576010605, 0.0074730846360555, 0.0026155796226175 };
	#define SamplesG 9
#elif ClarityRadius == 2
	float sampleOffsets[14] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130, 9.0869565, 11, 12.9130435, 14.826087, 16.7391305, 18.652174, 20.5652175, 22.478261, 24.3913045 };
	float sampleWeights[14] = { 0.07217787046621, 0.0739383063308, 0.07217787046621, 0.0671422050846, 0.0595124090525, 0.050254923194, 0.0404224382261, 0.030961867593, 0.0225763617753, 0.015665230617, 0.01033905220881, 0.006487248442, 0.0038673981121, 0.0021890932698 };
	#define SamplesG 14
#elif ClarityRadius == 3
	float sampleOffsets[49] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130, 9.0869565, 11, 12.9130435, 14.826087, 16.7391305, 18.652174, 20.5652175, 22.478261, 24.3913045, 26.304348, 28.2173915, 30.130435, 32.0434785,  33.956522, 35.8695655, 37.782609, 39.6956525, 41.608696, 43.5217395, 45.434783, 47.3478265, 49.26087, 51.1739135, 53.086957, 55.0000005, 56.913044, 58.8260875, 60.739131, 62.6521745, 62.6521745, 66.4782615, 68.391305, 70.3043485, 72.217392, 74.1304355, 76.043479, 77.9565225, 79.869566, 81.7826095, 83.695653, 85.6086965, 87.52174, 89.4347835, 91.347827 };
	float sampleWeights[27] = { 0.038588721157425, 0.04019658453898,  0.0393926528482025, 0.038588721157425, 0.0370754379748375, 0.03556215479225, 0.033510492016125, 0.03145882924, 0.02908457797625, 0.0267103267125, 0.02423714831875, 0.021763969925, 0.01938971865625, 0.0170154673875, 0.01488853396875, 0.01276160055, 0.0109704987125, 0.009179396875, 0.0077550077065, 0.006330618538, 0.00525763234525, 0.0041846461525, 0.0034174610244375, 0.002650275896375, 0.0021289101456875, 0.001607544395, 0.001270478635 };
	#define SamplesG 27
#elif ClarityRadius >= 4
	float sampleOffsets[49] = { 0.0, 1.4347826, 3.3478260, 5.2608695, 7.1739130, 9.0869565, 11, 12.9130435, 14.826087, 16.7391305, 18.652174, 20.5652175, 22.478261, 24.3913045, 26.304348, 28.2173915, 30.130435, 32.0434785,  33.956522, 35.8695655, 37.782609, 39.6956525, 41.608696, 43.5217395, 45.434783, 47.3478265, 49.26087, 51.1739135, 53.086957, 55.0000005, 56.913044, 58.8260875, 60.739131, 62.6521745, 62.6521745, 66.4782615, 68.391305, 70.3043485, 72.217392, 74.1304355, 76.043479, 77.9565225, 79.869566, 81.7826095, 83.695653, 85.6086965, 87.52174, 89.4347835, 91.347827 };
	float sampleWeights[53] = { 0.03979461869359125*0.533, 0.04019658453898*0.533,  0.03979461869359125*0.533, 0.0393926528482025*0.533, 0.03899068700281375*0.533, 0.038588721157425*0.533, 0.03783207956613125*0.533, 0.0370754379748375*0.533, 0.03631879638354375*0.533, 0.03556215479225*0.533, 0.0345363234041875*0.533, 0.033510492016125*0.533, 0.0324846606280625*0.533, 0.03145882924*0.533, 0.030271703608125*0.533, 0.02908457797625*0.533, 0.027897452344375*0.533, 0.0267103267125*0.533, 0.025473737515625*0.533, 0.02423714831875*0.533, 0.023000559121875*0.533, 0.021763969925*0.533, 0.020576844290625*0.533, 0.01938971865625*0.533, 0.018202593021875*0.533, 0.0170154673875*0.533, 0.015952000678125*0.533, 0.01488853396875*0.533, 0.013825067259375*0.533, 0.01276160055*0.533, 0.01186604963125*0.533, 0.0109704987125*0.533, 0.01007494779375*0.533, 0.009179396875*0.533, 0.00846720229075*0.533, 0.0077550077065*0.533, 0.00704281312225*0.533, 0.006330618538*0.533, 0.005794125441625*0.533, 0.00525763234525*0.533, 0.004721139248875*0.533, 0.0041846461525*0.533, 0.00380105358846875*0.533, 0.0034174610244375*0.533, 0.00303386846040625*0.533, 0.002650275896375*0.533, 0.00238959302103125*0.533, 0.0021289101456875*0.533, 0.00186822727034375*0.533, 0.001607544395*0.533, 0.001439011515*0.533, 0.001270478635 };
	#define SamplesG 49
#endif

	float4 color = inputColor.Sample(colorLinearSampler, IN.texcoord.xy) * sampleWeights[0];
	[loop]
	for(int i = 1; i < SamplesG; ++i) {
		color += inputColor.Sample(colorLinearSampler, IN.texcoord.xy + float2(sampleOffsets[i] * cePX_SIZE.x, 0.0)) * sampleWeights[i];
		color += inputColor.Sample(colorLinearSampler, IN.texcoord.xy - float2(sampleOffsets[i] * cePX_SIZE.x, 0.0)) * sampleWeights[i]; 
	}
	
#if LightnessTest == 1
	color.r = max(color.r,max(color.g,color.b));
#elif LightnessTest == 2
	color.r = min(color.r,min(color.g,color.b));
#elif LightnessTest == 3 
	color.r = (max(color.r,max(color.g,color.b))+min(color.r,min(color.g,color.b)))*0.5;
#elif LightnessTest == 4
	color.rgb = argb2hsv(color.rgb);
	color.r = color.b;
#else
	color.r = dot(color.rgb,CoefLuma_HP);
#endif
	
	
	return color;
}
