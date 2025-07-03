!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
MOVH H1.xyz, f[TEX1];
DP3H H1.w, H1, H1;
LG2H H1.w, |H1.w|;
MULH H1.w, H1, {0.5, 0, 0, 0}.x; 
EX2H H1.w, -H1.w;
MULH H1.xyz, H1, H1.w;
DP3H o[COLH], H0, H1;
END

# Passes = 2 

# Registers = 1 

# Textures = 2 
