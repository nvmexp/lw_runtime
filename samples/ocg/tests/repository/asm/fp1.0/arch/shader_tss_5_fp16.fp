!!FP1.0 
# Pixelshader 005
# TSS count 4
TXP H0, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, LWBE;
MULH H2, H0, H1; # color & alpha paired
TXP H0, f[TEX2], TEX2, LWBE; # eliminated a MOV
TXP H1, f[TEX3], TEX3, 2D;
MADH H0, H0, H2, H1; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 4 
