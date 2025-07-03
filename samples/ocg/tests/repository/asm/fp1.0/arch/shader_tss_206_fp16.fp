!!FP1.0 
# Pixelshader 206
# TSS count 8
TXP H1, f[TEX0], TEX0, 2D;
MULH H0.xyz, H1, f[COL0];
MOVH H0.w, H1;
TXP H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H1, H0;
# alpha disabled
TXP H1, f[TEX2], TEX2, 2D;
MULH H0.xyz, H1, H0;
# alpha disabled
TXP H1, f[TEX3], TEX3, 2D;
MULH H0.xyz, H1, H0;
# alpha disabled
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 1 

# Textures = 4 
