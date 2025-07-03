!!FP1.0
DP3H H0.w, f[TEX0], f[TEX0];
LG2H H0.x, |H0.w|;
MULH H0.x, H0, {0.5, 0, 0, 0}.x; 
EX2H H0.x, -H0.x;
MULH H0, H0.x, H0.w;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 1 

# Textures = 1 
