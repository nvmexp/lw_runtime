!!FP2.0
DIVH H1.xyz, f[TEX1], f[TEX1].w;
MOVH H0.w, {1.987654, 1.788889, 1.610000, 1.449000}.x;
TEX H0, f[TEX0], TEX0, 2D;
DP3H H2.x, H1, H0;
DIVH H1.xyz, f[TEX2], f[TEX2].w;
DP3H H2.y, H1, H0;
TEX H0, H2, TEX2, 2D;
MULH H0.xyz, H0, {1.304100, 1.173690, 1.056321, 0.950689}.x;
END

# Passes = 6 

# Registers = 2 

# Textures = 3 
