!!FP1.0 
# Pixelshader 232
# TSS count 2
TXP H0, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H1, H0;
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 1 

# Textures = 2 
