!!FP1.0 
# Pixelshader 141
# TSS count 4
TXP H0, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
ADDH H3.xyz, H1, -H0;
MADH H0.xyz, H3, H0.w, H0;
MULH H0.xyz, f[COL0], H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
TXP H0.w, f[TEX3], TEX3, 2D; # eliminated a MOV
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 3 
