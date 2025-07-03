cbuffer ControlBuf
{
    bool g_chkLUT;//: chkLUT
	float g_sldLUT;
    bool g_chkCartoon;//: chkCartoon
	float g_sldCartoon;
    bool g_chkLevels;//: chkLevels
    float g_sldLevelsBlack;
    float g_sldLevelsWhite;
    bool g_chkSwfxTechnicolor;//: chkSwfxTechnicolor
    bool g_chkColorMood;//: chkColorMood
    bool g_chkFilmic;//: chkFilmic
    bool g_chkHueFX;//: chkHueFX
	float g_sldHueFXMid;
	float g_sldHueFXRange;
    bool g_chkLensDirt;//: chkLensDirt
    bool g_chkLensFlare;//: chkLensFlare
    bool g_chkSharpening;//: chkSharpening
    bool g_chkExplosion;//: chkExplosion
    bool g_chkHeatHaze;//: chkHeatHaze
    bool g_chkLED;//: chkLED
	float g_sldLEDRad;
	float g_sldLEDSize;
    bool g_chkLetterbox;//: chkLetterbox
}

// Defining twhat was previously set by UI controls
#define g_sldBrightness 0.0
#define g_sldContrast 0.0
#define g_sldGamma 0.0
#define g_chkAnamFlare 0