!!FP1.0 
# Pixelshader 140
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H0.xyz, H1, f[COL0];
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MULH H0.w, H1, f[COL0];
TXP H0.w, f[TEX1], TEX1, 2D; # eliminated a MOV
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 1 

# Textures = 2 
