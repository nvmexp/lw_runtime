float4 AnselColor(float4 colorInput)
{
	float4 clr = colorInput;
	float4 clr_out = float4(0.5954f * clr.r, 0.58f * clr.g, 0.3487f * clr.b, clr.a);

	float4 clr2 = clr * clr;
    clr_out += float4(-1.492f * clr2.r, -3.916f * clr2.g, -1.835f * clr2.b, 0.0f);

	float4 clr3 = clr2 * clr;
    clr_out += float4(19.17f * clr3.r, 26.03f * clr3.g, 12.92f * clr3.b, 0.0f);

	float4 clr4 = clr2 * clr2;
    clr_out += float4(-45.04f * clr4.r, -50.52 * clr4.g, -19.09 * clr4.b, 0.0f);
	
	float4 clr5 = clr3 * clr2;
    clr_out += float4(41.23f * clr5.r, 41.09 * clr5.g, 9.679 * clr5.b, 0.0f);

	float4 clr6 = clr4 * clr2;
    clr_out += float4(-13.47f * clr6.r, -12.28 * clr6.g, -1.066 * clr6.b, 0.0f);
		
	return clr_out;
}

float4 AnselSketch(float4 colorInput, float2 texCoords, Texture2D texColor, SamplerState sampState)
{
	const int ks = 3;
	// Sobel Horizontal
	float filterKernelH[ks * ks] =
			{
				 -1,  0,  1,
				 -2,  0,  2,
				 -1,  0,  1
			};
	// Sobel Vertical
	float filterKernelV[ks * ks] =
			{
				 -1, -2, -1,
				  0,  0,  0,
				  1,  2,  1
			};

	float4 clrH = { 0.0f, 0.0f, 0.0f, 0.0f };
	float4 clrV = { 0.0f, 0.0f, 0.0f, 0.0f };
	float4 clrOriginal;

	[unroll]
	for (int i = 0; i < ks; ++i)
	{
		[unroll]
		for (int j = 0; j < ks; ++j)  
		{
			float4 clr = texColor.Sample(sampState, texCoords, int2(i - ks/2, j - ks/2));
			
			if (i == ks/2 && j == ks/2)
				clrOriginal = clr;
			
			clrH += filterKernelH[i+j*ks] * clr;
			clrV += filterKernelV[i+j*ks] * clr;
		}
	}

	// BW result
//	const float4 lumFilter = { 0.2126, 0.7152, 0.0722, 0.0 };
//	return float4( (1.0 - length(float2( dot(clrH, lumFilter), dot(clrV, lumFilter) ))).xxx, 1.0 ); 

#define ILWERT

	float3 sobelLengths =
#ifndef ILWERT
		{
			length( float2(clrH.r, clrV.r) ),
			length( float2(clrH.g, clrV.g) ),
			length( float2(clrH.b, clrV.b) )
		};
#else
		{
			1.0 - length( float2(clrH.r, clrV.r) ),
			1.0 - length( float2(clrH.g, clrV.g) ),
			1.0 - length( float2(clrH.b, clrV.b) )
		};
#endif
	return float4( lerp(colorInput.rgb, sobelLengths, 0.45), colorInput.a ); 
}

float3 CartoonPass( float3 colorInput, float2 tex, float2 pixelsize, Texture2D texColor, SamplerState sampState)
{
 
  	float diff1 = dot(LumCoeff, texColor.Sample(sampState, tex + pixelsize).rgb);
  	diff1 = dot(float4(LumCoeff,-1.0),float4( texColor.Sample(sampState, tex - pixelsize).rgb , diff1));
  
  	float diff2 = dot(LumCoeff, texColor.Sample(sampState, tex +float2(pixelsize.x,-pixelsize.y)).rgb);
  	diff2 = dot(float4(LumCoeff,-1.0),float4( texColor.Sample(sampState, tex +float2(-pixelsize.x,pixelsize.y)).rgb , diff2));
    
  	float edge = dot(float2(diff1,diff2),float2(diff1,diff2));
  
  	colorInput.rgb =  pow(edge,CartoonEdgeSlope) * -CartoonPower + colorInput.rgb;
	
	return saturate(colorInput);
}

float3 LevelsPass( float3 colorInput )
{
	#define black_point_float ( Levels_black_point / 255.0 )
	#define white_point_float ( 255.0 / (Levels_white_point - Levels_black_point)) 

 	colorInput.rgb = colorInput.rgb * white_point_float - (black_point_float *  white_point_float);
  	return colorInput;
}

float3 TechniPass_prod80(float3 colorInput)
{

	float3 colStrength = float3(ColStrengthR,ColStrengthG,ColStrengthB);
	float3 tsource = saturate(colorInput.rgb);
	float3 ttemp = 1 - tsource;
	float3 ttarget = ttemp.grg;
	float3 ttarget2 = ttemp.bbr;
	float3 ttemp2 = tsource.rgb * ttarget.rgb;
	ttemp2.rgb *= ttarget2.rgb;

	ttemp.rgb = ttemp2.rgb * colStrength;
	ttemp2.rgb *= TechniBrightness;

	ttarget.rgb = ttemp.grg;
	ttarget2.rgb = ttemp.bbr;

	ttemp.rgb = tsource.rgb - ttarget.rgb;
	ttemp.rgb += ttemp2.rgb;
	ttemp2.rgb = ttemp.rgb - ttarget2.rgb;

	colorInput.rgb = lerp(tsource.rgb, ttemp2.rgb, TechniStrength);

	colorInput.rgb = lerp(dot(colorInput.rgb, 0.333), colorInput.rgb, TechniSat); 
	
	return colorInput.rgb;

}

float3 TechnicolorPass( float3 colorInput )
{

	#define cyanfilter float3(0.0, 1.30, 1.0)
	#define magentafilter float3(1.0, 0.0, 1.05) 
	#define yellowfilter float3(1.6, 1.6, 0.05)

	#define redorangefilter float2(1.05, 0.620) //RG_
	#define greenfilter float2(0.30, 1.0)       //RG_
	#define magentafilter2 magentafilter.rb     //R_B

	float3 tcol = colorInput.rgb;
	
  	float2 rednegative_mul   = tcol.rg * (1.0 / (redNegativeAmount * TechniPower));
	float2 greennegative_mul = tcol.rg * (1.0 / (greenNegativeAmount * TechniPower));
	float2 bluenegative_mul  = tcol.rb * (1.0 / (blueNegativeAmount * TechniPower));
	
  	float rednegative   = dot( redorangefilter, rednegative_mul );
	float greennegative = dot( greenfilter, greennegative_mul );
	float bluenegative  = dot( magentafilter2, bluenegative_mul );
	
	float3 redoutput   = rednegative.rrr + cyanfilter;
	float3 greenoutput = greennegative.rrr + magentafilter;
	float3 blueoutput  = bluenegative.rrr + yellowfilter;
	
	float3 result = redoutput * greenoutput * blueoutput;
	colorInput.rgb = lerp(tcol, result, TechniAmount);
	return colorInput;
}

float3 DPXPass(float3 InputColor)
{
	float3x3 RGB =
	float3x3(
	2.67147117265996,-1.26723605786241,-0.410995602172227,
	-1.02510702934664,1.98409116241089,0.0439502493584124,
	0.0610009456429445,-0.223670750812863,1.15902104167061
	);

	float3x3 XYZ =
	float3x3(
	0.500303383543316,0.338097573222739,0.164589779545857,
	0.257968894274758,0.676195259144706,0.0658358459823868,
	0.0234517888692628,0.1126992737203,0.866839673124201
	);

	float DPXContrast = 0.1;
	float DPXGamma = 1.0;

	float RedLwrve = DPXRed;
	float GreenLwrve = DPXGreen;
	float BlueLwrve = DPXBlue;
	
	float3 RGB_Lwrve = float3(DPXRed,DPXGreen,DPXBlue);
	float3 RGB_C = float3(DPXRedC,DPXGreenC,DPXBlueC);

	float3 B = InputColor.rgb;
	B = pow(B, 1.0/DPXGamma);
 	B = B * (1.0 - DPXContrast) + (0.5 * DPXContrast);

	float3 Btemp = (1.0 / (1.0 + exp(RGB_Lwrve / 2.0)));	  
	B = ((1.0 / (1.0 + exp(-RGB_Lwrve * (B - RGB_C)))) / (-2.0 * Btemp + 1.0)) + (-Btemp / (-2.0 * Btemp + 1.0));

    	float value = max(max(B.r, B.g), B.b);
	float3 color = B / value;
	color = saturate(color);
	color = pow(color, 1.0/DPXColorGamma);
	
	float3 c0 = color * value;
        c0 = mul(XYZ, c0);

	float luma = dot(c0, float3(0.30, 0.59, 0.11)); //Use BT 709 instead?
        c0 = (1.0 - DPXSaturation) * luma + DPXSaturation * c0;
	c0 = mul(RGB, c0);
	
	InputColor.rgb = lerp(InputColor.rgb, c0, DPXBlend);

	return InputColor;
}

float3 LiftGammaGainPass( float3 colorInput )
{
	// -- Get input --
	float3 color = colorInput.rgb;
	
	// -- Lift --
	color = color * (1.5-0.5 * RGB_Lift) + 0.5 * RGB_Lift - 0.5;
	color = saturate(color); //isn't strictly necessary, but doesn't cost performance.
	
	// -- Gain --
	color *= RGB_Gain; 
	
	// -- Gamma --
	colorInput.rgb = pow(color, 1.0 / RGB_Gamma); //Gamma
	
	// -- Return output --
	//return (colorInput);
	return saturate(colorInput);
}

float3 TonemapPass( float3 colorInput )
{
	float3 color = colorInput.rgb;

	color = saturate(color - Defog * FogColor); // Defog
	
	color *= pow(2.0f, Exposure); // Exposure
	
	color = pow(color, Gamma);    // Gamma -- roll into the first gamma correction in main.h ?
	
	float lum = dot(LumCoeff, color.rgb);
	
	float3 blend = lum.rrr; //dont use float3
	
	float L = saturate( 10.0 * (lum - 0.45) );
  	
	float3 result1 = 2.0f * color.rgb * blend;
	float3 result2 = 1.0f - 2.0f * (1.0f - blend) * (1.0f - color.rgb);
	
	float3 newColor = lerp(result1, result2, L);
	float3 A2 = Bleach * color.rgb; //why use a float for A2 here and then multiply by color.rgb (a float3)?
	float3 mixRGB = A2 * newColor;
	
	color.rgb += ((1.0f - A2) * mixRGB);
	
	float3 middlegray = dot(color,(1.0/3.0)); //1fps slower than the original on lwpu, 2 fps faster on AMD
	
	float3 diffcolor = color - middlegray; //float 3 here
	colorInput.rgb = (color + diffcolor * Saturation)/(1+(diffcolor*Saturation)); //saturation
	
	return colorInput;
}

float3 VibrancePass( float3 colorInput )
{
   	#define Vibrance_coeff float3(Vibrance_RGB_balance * Vibrance)

	float3 color = colorInput; //original input color
  	float3 lumCoeff = float3(0.212656, 0.715158, 0.072186);  //Values to callwlate luma with

	float luma = dot(LumCoeff, color.rgb); //callwlate luma (grey)

	float max_color = max(colorInput.r, max(colorInput.g,colorInput.b)); //Find the strongest color
	float min_color = min(colorInput.r, min(colorInput.g,colorInput.b)); //Find the weakest color

  	float color_saturation = max_color - min_color; //The difference between the two is the saturation

   	color.rgb = lerp(luma, color.rgb, (1.0 + (Vibrance_coeff * (1.0 - (sign(Vibrance_coeff) * color_saturation))))); //extrapolate between luma and original by 1 + (1-saturation) - current

 	return color; //return the result
}

float3 LwrvesPass( float3 colorInput )
{
  float Lwrves_contrast_blend = Lwrves_contrast;


   /*-----------------------------------------------------------.
  /               Separation of Luma and Chroma                 /
  '-----------------------------------------------------------*/

  	// -- Callwlate Luma and Chroma if needed --
  	#if Lwrves_mode != 2

    	//callwlate luma (grey)
    	float luma = dot(LumCoeff, colorInput.rgb);

    	//callwlate chroma
	float3 chroma = colorInput.rgb - luma;
  	#endif

  	// -- Which value to put through the contrast formula? --
  	// I name it x because makes it easier to copy-paste to Graphtoy or Wolfram Alpha or another graphing program
  	#if Lwrves_mode == 2
	float3 x = colorInput.rgb; //if the lwrve should be applied to both Luma and Chroma
	#elif Lwrves_mode == 1
	float3 x = chroma; //if the lwrve should be applied to Chroma
	x = x * 0.5 + 0.5; //adjust range of Chroma from -1 -> 1 to 0 -> 1
  	#else // Lwrves_mode == 0
    	float x = luma; //if the lwrve should be applied to Luma
  	#endif

   /*-----------------------------------------------------------.
  /                     Contrast formulas                       /
  '-----------------------------------------------------------*/

  	// -- Lwrve 1 --
  	#if Lwrves_formula == 1
    	x = sin(PI * 0.5 * x); // Sin - 721 amd fps, +vign 536 lw
    	x *= x;
    
    	//x = 0.5 - 0.5*cos(PI*x);
    	//x = 0.5 * -sin(PI * -x + (PI*0.5)) + 0.5;
  	#endif

  	// -- Lwrve 2 --
  	#if Lwrves_formula == 2
    	x = x - 0.5;  
    	x = ( x / (0.5 + abs(x)) ) + 0.5;
    
    	//x = ( (x - 0.5) / (0.5 + abs(x-0.5)) ) + 0.5;
  	#endif

  	// -- Lwrve 3 --
  	#if Lwrves_formula == 3
    	//x = smoothstep(0.0,1.0,x); //smoothstep
    	x = x*x*(3.0-2.0*x); //faster smoothstep alternative - 776 amd fps, +vign 536 lw
    	//x = x - 2.0 * (x - 1.0) * x* (x- 0.5);  //2.0 is contrast. Range is 0.0 to 2.0
  	#endif

  	// -- Lwrve 4 --
  	#if Lwrves_formula == 4
    	x = (1.0524 * exp(6.0 * x) - 1.05248) / (20.0855 + exp(6.0 * x)); //exp formula
  	#endif

  	// -- Lwrve 5 --
  	#if Lwrves_formula == 5
    	//x = 0.5 * (x + 3.0 * x * x - 2.0 * x * x * x); //a simplified catmull-rom (0,0,1,1) - btw smoothstep can also be expressed as a simplified catmull-rom using (1,0,1,0)
    	//x = (0.5 * x) + (1.5 -x) * x*x; //estrin form - faster version
    	x = x * (x * (1.5-x) + 0.5); //horner form - fastest version

    	Lwrves_contrast_blend = Lwrves_contrast * 2.0; //I multiply by two to give it a strength closer to the other lwrves.
  	#endif

 	// -- Lwrve 6 --
  	#if Lwrves_formula == 6
    	x = x*x*x*(x*(x*6.0 - 15.0) + 10.0); //Perlins smootherstep
  	#endif

	// -- Lwrve 7 --
  	#if Lwrves_formula == 7
    	//x = ((x-0.5) / ((0.5/(4.0/3.0)) + abs((x-0.5)*1.25))) + 0.5;
	x = x - 0.5;
	x = x / ((abs(x)*1.25) + 0.375 ) + 0.5;
	//x = ( (x-0.5) / ((abs(x-0.5)*1.25) + (0.5/(4.0/3.0))) ) + 0.5;
  	#endif

  	// -- Lwrve 8 --
  	#if Lwrves_formula == 8
    	x = (x * (x * (x * (x * (x * (x * (1.6 * x - 7.2) + 10.8) - 4.2) - 3.6) + 2.7) - 1.8) + 2.7) * x * x; //Techicolor Cinestyle - almost identical to lwrve 1
  	#endif

  	// -- Lwrve 9 --
  	#if Lwrves_formula == 9
    	x =  -0.5 * (x*2.0-1.0) * (abs(x*2.0-1.0)-2.0) + 0.5; //parabola
  	#endif

  	// -- Lwrve 10 --
  	#if Lwrves_formula == 10 //Half-circles

    	#if Lwrves_mode == 0
      	float xstep = step(x,0.5);
	float xstep_shift = (xstep - 0.5);
	float shifted_x = x + xstep_shift;
   	#else
      	float3 xstep = step(x,0.5);
	float3 xstep_shift = (xstep - 0.5);
	float3 shifted_x = x + xstep_shift;
    	#endif

	x = abs(xstep - sqrt(-shifted_x * shifted_x + shifted_x) ) - xstep_shift;

  	//x = abs(step(x,0.5)-sqrt(-(x+step(x,0.5)-0.5)*(x+step(x,0.5)-0.5)+(x+step(x,0.5)-0.5)))-(step(x,0.5)-0.5); //single line version of the above
    
  	//x = 0.5 + (sign(x-0.5)) * sqrt(0.25-(x-trunc(x*2))*(x-trunc(x*2))); //worse
  
  	/* // if/else - even worse
  	if (x-0.5)
  	x = 0.5-sqrt(0.25-x*x);
  	else
  	x = 0.5+sqrt(0.25-(x-1)*(x-1));
	*/

  	//x = (abs(step(0.5,x)-clamp( 1-sqrt(1-abs(step(0.5,x)- frac(x*2%1)) * abs(step(0.5,x)- frac(x*2%1))),0 ,1))+ step(0.5,x) )*0.5; //worst so far
	
	//TODO: Check if I could use an abs split instead of step. It might be more efficient
	
	Lwrves_contrast_blend = Lwrves_contrast * 0.5; //I divide by two to give it a strength closer to the other lwrves.
  	#endif

  	// -- Lwrve 11 --
  	#if Lwrves_formula == 11 //Cubic catmull
    	float a = 1.00; //control point 1
    	float b = 0.00; //start point
    	float c = 1.00; //endpoint
    	float d = 0.20; //control point 2
    	x = 0.5 * ((-a + 3*b -3*c + d)*x*x*x + (2*a -5*b + 4*c - d)*x*x + (-a+c)*x + 2*b); //A lwstomizable cubic catmull-rom spline
  	#endif

  	// -- Lwrve 12 --
  	#if Lwrves_formula == 12 //Cubic Bezier spline
    	float a = 0.00; //start point
    	float b = 0.00; //control point 1
    	float c = 1.00; //control point 2
    	float d = 1.00; //endpoint

    	float r  = (1-x);
	float r2 = r*r;
	float r3 = r2 * r;
	float x2 = x*x;
	float x3 = x2*x;
	//x = dot(float4(a,b,c,d),float4(r3,3*r2*x,3*r*x2,x3));

	//x = a * r*r*r + r * (3 * b * r * x + 3 * c * x*x) + d * x*x*x;
	//x = a*(1-x)*(1-x)*(1-x) +(1-x) * (3*b * (1-x) * x + 3 * c * x*x) + d * x*x*x;
	x = a*(1-x)*(1-x)*(1-x) + 3*b*(1-x)*(1-x)*x + 3*c*(1-x)*x*x + d*x*x*x;
  	#endif

  	// -- Lwrve 13 --
  	#if Lwrves_formula == 13 //Cubic Bezier spline - alternative implementation.
    	float3 a = float3(0.00,0.00,0.00); //start point
    	float3 b = float3(0.25,0.15,0.85); //control point 1
    	float3 c = float3(0.75,0.85,0.15); //control point 2
    	float3 d = float3(1.00,1.00,1.00); //endpoint

    	float3 ab = lerp(a,b,x);           // point between a and b
    	float3 bc = lerp(b,c,x);           // point between b and c
    	float3 cd = lerp(c,d,x);           // point between c and d
    	float3 abbc = lerp(ab,bc,x);       // point between ab and bc
    	float3 bccd = lerp(bc,cd,x);       // point between bc and cd
    	float3 dest = lerp(abbc,bccd,x);   // point on the bezier-lwrve
    	x = dest;
  	#endif

  	// -- Lwrve 14 --
  	#if Lwrves_formula == 14
    	x = 1.0 / (1.0 + exp(-(x * 10.0 - 5.0))); //alternative exp formula
  	#endif

   /*-----------------------------------------------------------.
  /                 Joining of Luma and Chroma                  /
  '-----------------------------------------------------------*/

  	#if Lwrves_mode == 2 //Both Luma and Chroma
	float3 color = x;  //if the lwrve should be applied to both Luma and Chroma
	colorInput.rgb = lerp(colorInput.rgb, color, Lwrves_contrast_blend); //Blend by Lwrves_contrast

  	#elif Lwrves_mode == 1 //Only Chroma
	x = x * 2.0 - 1.0; //adjust the Chroma range back to -1 -> 1
	float3 color = luma + x; //Luma + Chroma
	colorInput.rgb = lerp(colorInput.rgb, color, Lwrves_contrast_blend); //Blend by Lwrves_contrast

  	#else // Lwrves_mode == 0 //Only Luma
    	x = lerp(luma, x, Lwrves_contrast_blend); //Blend by Lwrves_contrast
    	colorInput.rgb = x + chroma; //Luma + Chroma

  	#endif

  	//Return the result
  	return colorInput;
}

float3 SepiaPass( float3 colorInput )
{
	float3 sepia = colorInput.rgb;
	
	// callwlating amounts of input, grey and sepia colors to blend and combine
	float grey = dot(sepia, LumCoeff);
	sepia *= ColorTone;
	
	float3 blend2 = (grey * GreyPower) + (colorInput.rgb / (GreyPower + 1));

	colorInput.rgb = lerp(blend2, sepia, SepiaPower);
	// returning the final color
	return colorInput;
}


float3 SkyrimTonemapPass( float3 color )
{
	float	grayadaptation = dot(color.xyz, LumCoeff);

	#if (POSTPROCESS==1)
	color.xyz =  color.xyz / (grayadaptation * EAdaptationMaxV1 + EAdaptationMilw1);
	float cgray = dot( color.xyz, LumCoeff);
	cgray = pow(cgray, EContrastV1);
	float3 poweredcolor = pow( abs(color.xyz), EColorSaturatiolw1);
	float newgray = dot(poweredcolor.xyz, LumCoeff);
	color.xyz = poweredcolor.xyz * cgray / (newgray + 0.0001);
	float3	luma =  color.xyz;
	float	lumamax = 300.0;
	color.xyz = ( color.xyz * (1.0 +  color.xyz / lumamax)) / ( color.xyz + EToneMappingLwrveV1);	
	#endif

	#if (POSTPROCESS==2)
	color.xyz =  color.xyz / (grayadaptation * EAdaptationMaxV2 + EAdaptationMilw2);
	float3 xncol = normalize( color.xyz);
	float3 scl =  color.xyz / xncol.xyz;
	scl = pow(scl, EIntensityContrastV2);
	xncol.xyz = pow(xncol.xyz, EColorSaturatiolw2);
	color.xyz = scl*xncol.xyz;
	float	lumamax = EToneMappingOversaturatiolw2;
	color.xyz = ( color.xyz * (1.0 +  color.xyz / lumamax)) / ( color.xyz + EToneMappingLwrveV2);
 	color.xyz*=4;
	#endif

	#if (POSTPROCESS==3)
	color.xyz *= 35;
	float	lumamax = EToneMappingOversaturatiolw3;
	color.xyz = ( color.xyz * (1.0 +  color.xyz / lumamax)) / ( color.xyz + EToneMappingLwrveV3);
	#endif

	#if (POSTPROCESS == 4)
	color.xyz =  color.xyz / (grayadaptation * EAdaptationMaxV4 + EAdaptationMilw4);
	float Y = dot( color.xyz, float3(0.299, 0.587, 0.114)); //0.299 * R + 0.587 * G + 0.114 * B;
	float U = dot( color.xyz, float3(-0.14713, -0.28886, 0.436)); //-0.14713 * R - 0.28886 * G + 0.436 * B;
	float V = dot( color.xyz, float3(0.615, -0.51499, -0.10001)); //0.615 * R - 0.51499 * G - 0.10001 * B;
	Y = pow(Y, EBrightnessLwrveV4);
	Y = Y * EBrightnessMultiplierV4;
	color.xyz = V * float3(1.13983, -0.58060, 0.0) + U * float3(0.0, -0.39465, 2.03211) + Y;
	color.xyz = max( color.xyz, 0.0);
	color.xyz =  color.xyz / ( color.xyz + EBrightnessToneMappingLwrveV4);
	#endif

	#if (POSTPROCESS == 5)
	float hnd = 1;
	float2 hndtweak = float2( 3.1 , 1.5 );
        color.xyz *= lerp( hndtweak.x, hndtweak.y, hnd );
	float3 xncol = normalize( color.xyz);
	float3 scl =  color.xyz/xncol.xyz;
	scl = pow(scl, EIntensityContrastV5);
	xncol.xyz = pow(xncol.xyz, EColorSaturatiolw5);
	color.xyz = scl*xncol.xyz;
	color.xyz *= HCompensateSatV5; // compensate for darkening caused my EcolorSat above
	color.xyz =  color.xyz / ( color.xyz + EToneMappingLwrveV5);
	color.xyz *= 4;
	#endif

	#if (POSTPROCESS==6)
	//Postprocessing V6 by Kermles
	//tuned by the master himself for ME 1.4, thanks man!!!
	//hd6/ppv2///////////////////////////////////////////
	float 	EIntensityContrastV6 = EIntensityContrastV6Day;
	float 	EColorSaturatiolw6 = EColorSaturatiolw6Day;
	float 	HCompensateSatV6 = HCompensateSatV6Day;
	float 	EToneMappingLwrveV6 = EToneMappingLwrveV6Day;
	float 	EBrightnessV6 = EBrightnessV6Day;
	float 	EToneMappingOversaturatiolw6 = EToneMappingOversaturatiolw6Day;
	float 	EAdaptationMaxV6 = EAdaptationMaxV6Day;
	float 	EAdaptationMilw6 = EAdaptationMilw6Day;
	float	lumamax = EToneMappingOversaturatiolw6;
	//kermles////////////////////////////////////////////
	float4 	ncolor;					//temporary variable for color adjustments		
	//begin pp code/////////////////////////////////////////////////
	//ppv2 modified by kermles//////////////////////////////////////
		
	grayadaptation = clamp(grayadaptation, 0, 50);
	color.xyz *= EBrightnessV6;
	float3 xncol = normalize( color.xyz);
	float3 scl =  color.xyz/xncol.xyz;
	scl = pow(saturate(scl), EIntensityContrastV6);
	xncol.xyz = pow(xncol.xyz, EColorSaturatiolw6);
	color.xyz = scl*xncol.xyz;
	color.xyz *= HCompensateSatV6;
	color.xyz = ( color.xyz * (1.0 +  color.xyz/lumamax))/( color.xyz + EToneMappingLwrveV6);
	color.xyz /= grayadaptation*EAdaptationMaxV6+EAdaptationMilw6;
	//rerun ppv2////////////////////////////////////////////////////
	color.xyz *= EBrightnessV6;
	xncol = normalize( color.xyz);
	scl =  color.xyz/xncol.xyz;
	scl = saturate(scl);
	scl = pow(scl, EIntensityContrastV6);
	xncol.xyz = pow(xncol.xyz, EColorSaturatiolw6);
	color.xyz = scl*xncol.xyz;
	color.xyz *= HCompensateSatV6;
	color.xyz = ( color.xyz * (1.0 +  color.xyz/lumamax))/( color.xyz + EToneMappingLwrveV6);
	#endif

	return color;

}

float3 MoodPass( float3 colorInput )
{
	float3 colInput = colorInput;
	float3 colMood = 1.0f;
	colMood.r = moodR;
	colMood.g = moodG;
	colMood.b = moodB;
	float fLum = ( colInput.r + colInput.g + colInput.b ) / 3;
	colMood = lerp(0, colMood, saturate(fLum * 2.0));
	colMood = lerp(colMood, 1, saturate(fLum - 0.5) * 2.0);
	float3 colOutput = lerp(colInput, colMood, saturate(fLum * fRatio));
	colorInput=max(0, colOutput);
	return colorInput;
}

float3 CrossPass(float3 color)
{
	float2 CrossMatrix [3] = {
		float2 (1.03, 0.04),
		float2 (1.09, 0.01),
		float2 (0.78, 0.13),
 		};

	float3 image1 = color;
	float3 image2 = color;
	float gray = dot(float3(0.5,0.5,0.5), image1);  
	image1 = lerp (gray, image1,CrossSaturation);
	image1 = lerp (0.35, image1,CrossContrast);
	image1 +=CrossBrightness;
	image2.r = image1.r * CrossMatrix[0].x + CrossMatrix[0].y;
	image2.g = image1.g * CrossMatrix[1].x + CrossMatrix[1].y;
	image2.b = image1.b * CrossMatrix[2].x + CrossMatrix[2].y;
	color = lerp(image1, image2, CrossAmount);
	return color;
}

float3 FilmPass(float3 B)
{
	float3 G = B;
	float3 H = 0.01;
 
	B = saturate(B);
	B = pow(B, Linearization);
	B = lerp(H, B, Contrast);
 
	float A = dot(B.rgb, LumCoeff);
	float3 D = A;
 
	B = pow(B, 1.0 / BaseGamma);
 
	float a = FRedLwrve;
	float b = FGreenLwrve;
	float c = FBlueLwrve;
	float d = BaseLwrve;
 
	float y = 1.0 / (1.0 + exp(a / 2.0));
	float z = 1.0 / (1.0 + exp(b / 2.0));
	float w = 1.0 / (1.0 + exp(c / 2.0));
	float v = 1.0 / (1.0 + exp(d / 2.0));
 
	float3 C = B;
 
	D.r = (1.0 / (1.0 + exp(-a * (D.r - 0.5))) - y) / (1.0 - 2.0 * y);
	D.g = (1.0 / (1.0 + exp(-b * (D.g - 0.5))) - z) / (1.0 - 2.0 * z);
	D.b = (1.0 / (1.0 + exp(-c * (D.b - 0.5))) - w) / (1.0 - 2.0 * w);
 
	D = pow(D, 1.0 / EffectGamma);
 
	float3 Di = 1.0 - D;
 
	D = lerp(D, Di, FBleach);
 
	D.r = pow(abs(D.r), 1.0 / EffectGammaR);
	D.g = pow(abs(D.g), 1.0 / EffectGammaG);
	D.b = pow(abs(D.b), 1.0 / EffectGammaB);
 
	if (D.r < 0.5)
		C.r = (2.0 * D.r - 1.0) * (B.r - B.r * B.r) + B.r;
	else
		C.r = (2.0 * D.r - 1.0) * (sqrt(B.r) - B.r) + B.r;
 
	if (D.g < 0.5)
		C.g = (2.0 * D.g - 1.0) * (B.g - B.g * B.g) + B.g;
	else
		C.g = (2.0 * D.g - 1.0) * (sqrt(B.g) - B.g) + B.g;
 	//if (AgainstAllAutority) 
	if (D.b < 0.5)
		C.b = (2.0 * D.b - 1.0) * (B.b - B.b * B.b) + B.b;
	else
		C.b = (2.0 * D.b - 1.0) * (sqrt(B.b) - B.b) + B.b;
 
	float3 F = lerp(B, C, Strenght);
 
	F = (1.0 / (1.0 + exp(-d * (F - 0.5))) - v) / (1.0 - 2.0 * v);
 
	float r2R = 1.0 - FSaturation;
	float g2R = 0.0 + FSaturation;
	float b2R = 0.0 + FSaturation;
 
	float r2G = 0.0 + FSaturation;
	float g2G = (1.0 - Fade) - FSaturation;
	float b2G = (0.0 + Fade) + FSaturation;
 
	float r2B = 0.0 + FSaturation;
	float g2B = (0.0 + Fade) + FSaturation;
	float b2B = (1.0 - Fade) - FSaturation;
 
	float3 iF = F;
 
	F.r = (iF.r * r2R + iF.g * g2R + iF.b * b2R);
	F.g = (iF.r * r2G + iF.g * g2G + iF.b * b2G);
	F.b = (iF.r * r2B + iF.g * g2B + iF.b * b2B);
 
	float N = dot(F.rgb, LumCoeff);
	float3 Cn = F;
 
	if (N < 0.5)
		Cn = (2.0 * N - 1.0) * (F - F * F) + F;
	else
		Cn = (2.0 * N - 1.0) * (sqrt(F) - F) + F;
 
	Cn = pow(max(Cn,0), 1.0 / Linearization);
 
	float3 Fn = lerp(B, Cn, Strenght);
	return Fn;
}

float3 ReinhardToneMapping(in float3 x)
{
	const float W =  ReinhardWhitepoint;	// Linear White Point Value
    	const float K =  ReinhardScale;        // Scale

    	// gamma space or not?
    	return (1 + K * x / (W * W)) * x / (x + K);
}

float3 ReinhardLinearToneMapping(in float3 x)
{
    	const float W = ReinhardLinearWhitepoint;	        // Linear White Point Value
    	const float L = ReinhardLinearPoint;           // Linear point
    	const float C = ReinhardLinearSlope;           // Slope of the linear section
    	const float K = (1 - L * C) / C; // Scale (fixed so that the derivatives of the Reinhard and linear functions are the same at x = L)
    	float3 reinhard = L * C + (1 - L * C) * (1 + K * (x - L) / ((W - L) * (W - L))) * (x - L) / (x - L + K);

    	// gamma space or not?
    	return (x > L) ? reinhard : C * x;
}

float3 HaarmPeterDuikerFilmicToneMapping(in float3 x)
{
    	x = max( (float3)0.0f, x - 0.004f );
    	return pow( abs( ( x * ( 6.2f * x + 0.5f ) ) / ( x * ( 6.2f * x + 1.7f ) + 0.06 ) ), 2.2f );
}

float3 LwstomToneMapping(in float3 x)
{
	const float A = 0.665f;
	const float B = 0.09f;
	const float C = 0.004f;
	const float D = 0.445f;
	const float E = 0.26f;
	const float F = 0.025f;
	const float G = 0.16f;//0.145f;
	const float H = 1.1844f;//1.15f;

    // gamma space or not?
	return (((x*(A*x+B)+C)/(x*(D*x+E)+F))-G) / H;
}

float3 ColorFilmicToneMapping(in float3 x)
{
	// Filmic tone mapping
	const float3 A = float3(0.55f, 0.50f, 0.45f);	// Shoulder strength
	const float3 B = float3(0.30f, 0.27f, 0.22f);	// Linear strength
	const float3 C = float3(0.10f, 0.10f, 0.10f);	// Linear angle
	const float3 D = float3(0.10f, 0.07f, 0.03f);	// Toe strength
	const float3 E = float3(0.01f, 0.01f, 0.01f);	// Toe Numerator
	const float3 F = float3(0.30f, 0.30f, 0.30f);	// Toe Denominator
	const float3 W = float3(2.80f, 2.90f, 3.10f);	// Linear White Point Value
	const float3 F_linearWhite = ((W*(A*W+C*B)+D*E)/(W*(A*W+B)+D*F))-(E/F);
	float3 F_linearColor = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-(E/F);

    // gamma space or not?
	return pow(saturate(F_linearColor * 1.25 / F_linearWhite),1.25);
}

float3 ColormodPass( float3 color )
{
	float brightnessTerm = 2.0f * g_sldBrightness - 1.0f;
	
	const float contrastMin = -1.6094379124341003746007593332262; //ln(0.2)
	const float contrastMax = 1.6094379124341003746007593332262; //ln(5.0)
	
	float contrastFactor = exp(contrastMin  + g_sldContrast * (contrastMax  - contrastMin)); 
	//float contrastFactor = g_sldContrast <= 0.5f ? (0.5f - g_sldContrast) * -2.0f + 1.0f : (g_sldContrast - 0.5f) * 10.0f + 1.0f;

	color.xyz = (color.xyz - dot(color.xyz, 0.333)) * ColormodChroma + dot(color.xyz, 0.333);
	color.xyz = saturate(color.xyz);
	color.x = (pow(color.x, 2.0 * (1.0 - g_sldGamma) * ColormodGammaR) - 0.5) * ColormodContrastR * contrastFactor  + 0.5 + brightnessTerm;
	color.y = (pow(color.y, 2.0 * (1.0 - g_sldGamma) * ColormodGammaG) - 0.5) * ColormodContrastG * contrastFactor  + 0.5 + brightnessTerm;
	color.z = (pow(color.z, 2.0 * (1.0 - g_sldGamma) * ColormodGammaB) - 0.5) * ColormodContrastB * contrastFactor  + 0.5 + brightnessTerm;
	return color;	
}


float3 SphericalPass( float3 color )
{
	float3 signedColor = color.rgb * 2.0 - 1.0;
	float3 sphericalColor = sqrt(1.0 - signedColor.rgb * signedColor.rgb);
	sphericalColor = sphericalColor * 0.5 + 0.5;
	sphericalColor *= color.rgb;
	color.rgb += sphericalColor.rgb * sphericalAmount;
	color.rgb *= 0.95;
	return color;
}

float3 SincityPass(float3 color)
{
	float sinlumi = dot(color.rgb, float3(0.30f,0.59f,0.11f));
	if(color.r > (color.g + 0.2f) && color.r > (color.b + 0.025f))
	{
		color.rgb = float3(sinlumi, 0, 0)*1.5;
	}
	else
	{
		color.rgb = sinlumi;
	}
	return color;
}

float3 colorhuefx_prod80( float3 color )
{
	
	float3 fxcolor = saturate( color.xyz );
	float greyVal = dot( fxcolor.xyz, LumCoeff.xyz );
	float3 HueSat = Hue( fxcolor.xyz );
	float colorHue = HueSat.x;
	float colorInt = HueSat.z - HueSat.y * 0.5;
	float colorSat = HueSat.y / ( 1.0 - abs( colorInt * 2.0 - 1.0 ) * 1e-10 );

	//When color intensity not based on original saturation level
   	if ( USE_COLORSAT == 0 )   colorSat = 1.0f;

	float hueMin_1 = hueMid - hueRange;
	float hueMax_1 = hueMid + hueRange;
	float hueMin_2 = 0.0f;
	float hueMax_2 = 0.0f;


   	if ( hueMin_1 < 0.0 )
   	{
   		hueMin_2 = 1.0f + hueMin_1;
   		hueMax_2 = 1.0f + hueMid;
   
      		if ( colorHue >= hueMin_1 && colorHue <= hueMid )
         		fxcolor.xyz = lerp( greyVal.xxx, fxcolor.xyz, smootherstep( hueMin_1, hueMid, colorHue ) * ( colorSat * satLimit ));
      		else if ( colorHue >= hueMid && colorHue <= hueMax_1 )
        		fxcolor.xyz = lerp( greyVal.xxx, fxcolor.xyz, ( 1.0f - smootherstep( hueMid, hueMax_1, colorHue )) * ( colorSat * satLimit ));
      		else if ( colorHue >= hueMin_2 && colorHue <= hueMax_2 )
         		fxcolor.xyz = lerp( greyVal.xxx, fxcolor.xyz, smootherstep( hueMin_2, hueMax_2, colorHue ) * ( colorSat * satLimit ));
      		else
         		fxcolor.xyz = greyVal.xxx;
   	}

   	else if ( hueMax_1 > 1.0 )
   	{
   		hueMin_2 = 0.0f - ( 1.0f - hueMid );
   		hueMax_2 = hueMax_1 - 1.0f;

      		if ( colorHue >= hueMin_1 && colorHue <= hueMid )
         		fxcolor.xyz = lerp( greyVal.xxx, fxcolor.xyz, smootherstep( hueMin_1, hueMid, colorHue ) * ( colorSat * satLimit ));
      		else if ( colorHue >= hueMid && colorHue <= hueMax_1 )
         		fxcolor.xyz = lerp( greyVal.xxx, fxcolor.xyz, ( 1.0f - smootherstep( hueMid, hueMax_1, colorHue )) * ( colorSat * satLimit ));
      		else if ( colorHue >= hueMin_2 && colorHue <= hueMax_2 )
         		fxcolor.xyz = lerp( greyVal.xxx, fxcolor.xyz, ( 1.0f - smootherstep( hueMin_2, hueMax_2, colorHue )) * ( colorSat * satLimit ));
      		else
         		fxcolor.xyz = greyVal.xxx;
   	}	
   
	else
   	{
      		if ( colorHue >= hueMin_1 && colorHue <= hueMid )
        		fxcolor.xyz = lerp( greyVal.xxx, fxcolor.xyz, smootherstep( hueMin_1, hueMid, colorHue ) * ( colorSat * satLimit ));
      		else if ( colorHue > hueMid && colorHue <= hueMax_1 )
         		fxcolor.xyz = lerp( greyVal.xxx, fxcolor.xyz, ( 1.0f - smootherstep( hueMid, hueMax_1, colorHue )) * ( colorSat * satLimit ));
      		else
         		fxcolor.xyz = greyVal.xxx;
   	}

   	color.xyz = lerp( color.xyz, fxcolor.xyz, fxcolorMix );

	return color.xyz;

}
