   /*-----------------------------------------------------------.   
  /                    Cubic Lens Distortion                    /
  '-----------------------------------------------------------*/

    /*
            Cubic Lens Distortion HLSL Shader
           
            Original Lens Distortion Algorithm from SSontech (Syntheyes)
            http://www.ssontech.com/content/lensalg.htm
           
            r2 = image_aspect*image_aspect*u*u + v*v
            f = 1 + r2*(k + klwbe*sqrt(r2))
            u' = f*u
            v' = f*v
     
            author : Franois Tarlier
            website : http://www.francois-tarlier.com/blog/tag/lens-distortion/
     
    */
     
    float3 LensDistortionPass(float2 tex)
    {
           
            // lens distortion coefficient (between
            float k = -0.15;
           
            // cubic distortion value
            float klwbe = 0.5;
           
           
            float r2 = (tex.x-0.5) * (tex.x-0.5) + (tex.y-0.5) * (tex.y-0.5);       
            float f = 0.0;
           
     
            //only compute the cubic distortion if necessary
            if( klwbe == 0.0){
                    f = 1 + r2 * k;
            }else{
                    f = 1 + r2 * (k + klwbe * sqrt(r2));
            };
           
            // get the right pixel for the current position
            float x = f*(tex.x-0.5)+0.5;
            float y = f*(tex.y-0.5)+0.5;
            float3 inputDistord = tex2D(BorderSampler,float2(x,y)).rgb;
     
     
            return inputDistord;
    }



float3 LensDistortionWrap(float4 position : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
	float3 color = LensDistortionPass(texcoord).rgb;
	
	#ifdef Shared_Piggyback_LensDistortion
    color.rgb = SharedPass(texcoord, float4(color.rgbb)).rgb;
	#endif
	
	return color.rgb;
}

