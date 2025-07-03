!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
DP3H H0.w, H0, H0;
TEX H1, f[TEX0], TEX1, 2D;
LG2H H0.w, |H0.w|;
MULH H0.w, H0, {0.5, 0, 0, 0}.x; 
EX2H H0.w, -H0.w;
MADH H0.xyz, H0, -H0.w, -{1.304100, 1.173690, 1.056321, 0.950689}.x;
DP3H H0.w, H0, H0;
LG2H H0.w, |H0.w|;
MULH H0.w, H0, {0.5, 0, 0, 0}.x; 
EX2H H0.w, -H0.w;
MULH H0.xyz, H0, H0.w;
DP3H H1.w, H0, H1;
DP3H H1.x, -{1.304100, 1.173690, 1.056321, 0.950689}.x, H1;
TEX H0, H1.xwxx, TEX2, 2D;
MULH o[COLR], H0, {0.368316, 0.331485, 0.298336, 0.268503}.x;
END

# Passes = 10 

# Registers = 1 

# Textures = 1 
