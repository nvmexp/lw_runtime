!!FP2.0
DIVR R1.xyz, f[TEX1], f[TEX1].w;
MOVR R0.w, {1.987654, 1.788889, 1.610000, 1.449000}.x;
TEX R0, f[TEX0], TEX0, 2D;
DP3R R2.x, R1, R0;
DIVR R1.xyz, f[TEX2], f[TEX2].w;
DP3R R2.y, R1, R0;
TEX H0, R2, TEX2, 2D;
MULR H0.xyz, H0, {1.304100, 1.173690, 1.056321, 0.950689}.x;
END

# Passes = 6 

# Registers = 3 

# Textures = 3 
