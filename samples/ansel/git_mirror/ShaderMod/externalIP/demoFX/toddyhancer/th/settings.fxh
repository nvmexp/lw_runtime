//>Bloom Settings<
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if 0
// Original Bloom settings
#define iBloomMixmode			2	//[1 to 4] 1: Linear add | 2: Screen add | 3: Screen/Lighten/Opacity | 4: Lighten
#define fBloomThreshold			0.6	//[0.1 to 1.0] Every pixel brighter than this value triggers bloom.
#define fBloomAmount			0.3	//[1.0 to 20.0] Intensity of bloom.
#define fBloomSaturation 		0.8	//[0.0 to 2.0] Bloom saturation. 0.0 means white bloom, 2.0 means very very colorful bloom.
#define fBloomTint 		float3(0.7,0.8,1.0) //[0.0 to 1.0] R, G and B components of bloom tintcolor the bloom color gets shifted to.
#else
// Toddyhancer Bloom settings
#define iBloomMixmode 2 //[1|2|3|4] //-1 = Linear add | 2 = Screen add | 3 = Screen/Lighten/Opacity | 4 = Lighten
#define fBloomThreshold 0.0 //[0.1:1.0] //-Every pixel brighter than this value triggers bloom.
#define fBloomAmount 0.0001 //[0.0:20.0] //-Intensity of bloom.
#define fBloomSaturation 0.5 //[0.0:2.0] //-Bloom saturation. 0.0 means white bloom, 2.0 means very very colorful bloom.
#define fBloomTint float3(0.7,0.8,1.0) //[0.0:1.0] //-R, G and B components of bloom tintcolor the bloom color gets shifted to. X = Red, Y = Green, Z = Blue.
#endif

#define fAnamFlareThreshold		0.90	//[0.1 to 1.0] Every pixel brighter than this value gets a flare.
#define fAnamFlareWideness		2.4	//[1.0 to 2.5] Horizontal wideness of flare. Don't set too high, otherwise the single samples are visible
#define fAnamFlareAmount		14.5	//[1.0 to 20.0] Intensity of anamorphic flare.
#define fAnamFlareLwrve			1.2	//[1.0 to 2.0] Intensity lwrve of flare with distance from source
#define fAnamFlareColor		float3(0.012,0.313,0.588) //[0.0 to 1.0] R, G and B components of anamorphic flare. Flare is always same color.



//>Vibrance settings<
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define Vibrance 0.28 //[-1.00:1.00] //-Intelligently saturates (or desaturates if you use negative values) the pixels depending on their original saturation.
#define Vibrance_RGB_balance float3(1.00,1.00,1.00) //[-10.00:10.00] //-A per channel multiplier to the Vibrance strength so you can give more boost to certain colors over others. X = Red, Y = Green, Z = Blue



//>Vignette settings<
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define VignetteType 1 //[1|2|3] //-1 = Original, 2 = New, 3 = TV style
#define VignetteRatio 1.00 //[0.15:6.00] //-Sets a width to height ratio. 1.00 (1/1) is perfectly round, while 1.60 (16/10) is 60 % wider than it's high.
#define VignetteRadius 2.00 //[-1.00:3.00] //-lower values = stronger radial effect from center
#define VignetteAmount -1.00 //[-2.00:1.00] //-Strength of black. -2.00 = Max Black, 1.00 = Max White.
#define VignetteSlope 2 //[2:16] //-How far away from the center the change should start to really grow strong (odd numbers cause a larger fps drop than even numbers)
#define VignetteCenter float2(0.500,0.500) //[0.000:1.000] //-Center of effect for VignetteType 1. 2 and 3 do not obey this setting.



//>LumaSharpen settings<
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define sharp_strength 1.75 //[0.10:3.00] //-Strength of the sharpening
#define sharp_clamp 0.935 //[0.000:1.000] //-Limits maximum amount of sharpening a pixel recieves - Default is 0.035

//>Advanced sharpening settings<
#define pattern 2 //[1|2|3|4] //-Choose a sample pattern. 1 = Fast, 2 = Normal, 3 = Wider, 4 = Pyramid shaped.
#define offset_bias 7.0*g_sldLumaSharpenRad //[0.0:6.0] //-Offset bias adjusts the radius of the sampling pattern. I designed the pattern for offset_bias 1.0, but feel free to experiment.

//>Debug sharpening settings<
#define show_sharpen 0 //[0:1] //-Visualize the strength of the sharpen (multiplied by 4 to see it better)



//>Clarity Settings<
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define ClarityRadius 2 //[0:4] //-The radius of the effect. Higher values = a larger radius.
#define ClarityOffset 1.25 //[0.5:3.0] //-Values less than 1.0 will decrease the radius, values greater than 1.0 will increase the radius.
#define ClarityBlendMode 1 //[1|2|3|4|5|6|7] //-1 = Soft Light(weak), 2 = Overlay(neutral), 3 = Multiply, 4 = Hard Light, 5 = Vivid Light, 6 = Soft Light#2(lighter), 7 = Soft Light#3(darker)
#define ClarityStrength 0.40 //[-0.50:2.00] //-Strength of the effect. 

//>Advanced Clarity Settings<
#define DarkIntensity 1.5 //[0.0:4.0] //-Adjust the strength of dark halos.
#define LightIntensity 1.5 //[0.0:4.0] //-Adjust the strength of light halos.
#define ViewMask 0 //[0:1] //-The mask is what creates the effect. View it when making adjustments to get a better idea of how your changes will affect the image. 
#define LightnessTest 1 //[0:4] //-0 = BT.709 Luma, 1 = Max(R,G,B), 2 = Min(R,G,B), 3 = 1/2*Max(R,G,B) + 1/2*Min(R,G,B), 4 = Value channel in HSV (may cause some discoloration in dark spots).
#define BlendIfDark 0 //[0:255] //-Any pixels below this value will be excluded from the effect. Set to 50 to target mid-tones.
#define BlendIfLight 255 //[0:255] //-Any pixels above this value will be excluded from the effect. Set to 205 to target mid-tones.
#define ViewBlendIfMask 0 //[0:1] //-Areas covered in RED receive contrast enhancement, areas covered in BLUE do not. The stronger the color, the stronger the effect.

//>Performance and Misc Settings<
#define ClarityTextureFormat 0 //[0|1|2] //-0 = R8, 1 = R16F, 2 = R32F. 
#define ClarityTexScale 0 //[0|1|2] //-0 = no scaling, 1 = 1/2 resolution, 2 = 1/4 resolution. Reduces performance cost.  
#define Clarity_ToggleKey RFX_ToggleKey //[undef] //-Default is the "Insert" key. Change to RFX_ToggleKey to toggle with the rest of the Framework shaders.



//>ChromaticAberration<
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define CA_Color 1 //[1:4] //-1 = red/green/blue, 2 = red/cyan, 3 = blue/yellow, 4 = magenta/green
#define CA_Offset 0.0030 //[0.0000:0.0100] //-Adjusts the offset of the effect. Larger values = larger offset.
#define CA_Blurring 2 //[0:3] //-Blurs the CA making it smoother and wider. 0 = no blurring, 1 = 3 samples, 2 = 5 samples, 3 = 9 samples.
#define CA_Strength 0.40 //[0.00:2.00] //-Strength of the effect.
#define ca_ToggleKey RFX_ToggleKey //[undef] //-Default is the "Insert" key. Change to RFX_ToggleKey to toggle with the rest of the Framework shaders.

//>Lens Distortion Settings<
#define CA_RadialBlurring 2 //[0:4] //-Blurs the image outward from the center. 0 = off, 1 = 3 samples, 2 = 5 samples, 3 = 11 samples, 4 = 21 samples
#define RadialBlurStrength 1.0 //[0.0:1.0] //-Values less than 1.0 will reduce the amount of blurring to the image but the CA will remain blurred.
#define RadialBlurFolws 1.0 //[0.1:5.0] //-Determines how far from the center of the screen the RadialBlurring effect begins. Lower #'s = further from the center.
#define CA_LensShape 1.78 //[1.00:2.00] //-Adjusts the shape of the "Lens". 1.00 = Round, #'s > 1.00 = Ellipse.
#define CA_LensFolws 1.0 //[0.1:5.0] //-Determines how far from the center of the screen the CA effect begins. Lower #'s = further from the center.

