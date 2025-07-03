!!FP1.0 
# Pixelshader 223
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
DP3H H0.xyz, H1, f[COL1];
MULH H0.w, H1, f[COL1];
TXP H1, f[TEX1], TEX1, 2D;
MULH H0, H0, H1; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 1 

# Textures = 2 
