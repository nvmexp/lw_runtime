float3 HaarmPeterDuikerFilmicToneMapping(in float3 x)
{
   	x = max( (float3)0.0f, x - 0.004f );
   	return pow( abs( ( x * ( 6.2f * x + 0.5f ) ) / ( x * ( 6.2f * x + 1.7f ) + 0.06 ) ), 2.2f );
}
