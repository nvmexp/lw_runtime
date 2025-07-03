!!FP1.0 
# Pixelshader 231
# TSS count 4
TXP H0, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
ADDH H0.xyz, H1, H0;
TXP H1, f[TEX2], TEX2, 2D;
ADDH H0.xyz, H1, H0;
TXP H1, f[TEX3], TEX3, 2D;
ADDH H0.xyz, H1, H0;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 1 

# Textures = 4 
