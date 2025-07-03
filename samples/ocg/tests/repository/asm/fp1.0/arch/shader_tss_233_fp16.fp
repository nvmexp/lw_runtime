!!FP1.0 
# Pixelshader 233
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H0.xyz, H1, f[COL0];
MOVH H0.w, H1;
TXP H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H1, H0;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 1 

# Textures = 2 
