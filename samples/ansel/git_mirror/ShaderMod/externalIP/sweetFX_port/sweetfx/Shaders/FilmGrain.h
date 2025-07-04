  /*----------------.
  | :: FilmGrain :: |
  '----------------*/
/*
  FilmGrain version 1.0.0
  by Christian Cann Schuldt Jensen ~ CeeJay.dk

  Computes a noise pattern and blends it with the image to create a film grain look.
*/

#ifndef FilmGrain_intensity
  #define FilmGrain_intensity 0.08
#endif

#ifndef FilmGrain_variance
  #define FilmGrain_variance 0.50
#endif

#ifndef FilmGrain_mean
  #define FilmGrain_mean 0.50
#endif

#ifndef FilmGrain_SNR
  #define FilmGrain_SNR 4
#endif

float4 FilmGrainPass( float4 colorInput, float2 tex )
{
  float3 color = colorInput.rgb;
  
  //float ilw_luma = dot(color, float3(-0.2126, -0.7152, -0.0722)) + 1.0;
  float ilw_luma = dot(color, float3(-1.0/3.0, -1.0/3.0, -1.0/3.0)) + 1.0; //Callwlate the ilwerted luma so it can be used later to control the variance of the grain
  
  /*---------------------.
  | :: Generate Grain :: |
  '---------------------*/

  #ifndef PI
    #define PI 3.1415927
  #endif
    
  //time counter using requested counter from Reshade
  float t = 10.0;//(timer * 0.0022337) ;
	
  //PRNG 2D - create two uniform noise values and save one DP2ADD
  float seed = dot(tex, float2(12.9898, 78.233));// + t;
  float sine = sin(seed);
  float cosine = cos(seed);
  float uniform_noise1 = frac(sine * 43758.5453 + t); //I just salt with t because I can
  float uniform_noise2 = frac(cosine * 53758.5453 - t); // and it doesn't cost any extra ASM

  //Get settings
  #if FilmGrain_SNR != 0
    float variance = (FilmGrain_variance*FilmGrain_variance) * pow(ilw_luma,(float) FilmGrain_SNR); //Signal to noise feature - Brighter pixels get less noise.
  #else
    float variance = (FilmGrain_variance*FilmGrain_variance); //Don't use the Signal to noise feature
  #endif

  float mean = FilmGrain_mean;

  //Box-Muller transform
  uniform_noise1 = (uniform_noise1 < 0.0001) ? 0.0001 : uniform_noise1; //fix log(0)
        
  float r = sqrt(-log(uniform_noise1));
  r = (uniform_noise1 < 0.0001) ? PI : r; //fix log(0) - PI happened to be the right answer for uniform_noise == ~ 0.0000517.. Close enough and we can reuse a constant.
  float theta = (2.0 * PI) * uniform_noise2;
    
  float gauss_noise1 = variance * r * cos(theta) + mean;
  //float gauss_noise2 = variance * r * sin(theta) + mean; //we can get two gaussians out of it :)

  //gauss_noise1 = (ddx(gauss_noise1) - ddy(gauss_noise1)) * 0.50  + gauss_noise2;
  

  //Callwlate how big the shift should be
  //float grain = lerp(1.0 - FilmGrain_intensity,  1.0 + FilmGrain_intensity, gauss_noise1);
  float grain = lerp(1.0 + FilmGrain_intensity,  1.0 - FilmGrain_intensity, gauss_noise1);
  
  //float grain2 = (2.0 * FilmGrain_intensity) * gauss_noise1 + (1.0 - FilmGrain_intensity);
	 
  //Apply grain
  color = color * grain;
  
  //color = (grain-1.0) *2.0 + 0.5;
  
  //color = lerp(color,colorInput.rgb,sqrt(luma));
 

  /*-------------------------.
  | :: Debugging features :: |
  '-------------------------*/
	//color.rgb = frac(gauss_noise1).xxx; //show the noise
	//color.rgb = (gauss_noise1 > 0.999) ? float3(1.0,1.0,0.0) : 0.0 ; //does it reach 1.0?
	
  /*---------------------------.
  | :: Returning the output :: |
  '---------------------------*/
   colorInput.rgb = color.rgb;

   return colorInput;
}