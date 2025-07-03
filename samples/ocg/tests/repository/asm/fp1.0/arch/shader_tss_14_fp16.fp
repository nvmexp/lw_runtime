!!FP1.0 
# Pixelshader 014
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H0, f[COL0], H1; # color & alpha paired
TXP H1, f[TEX1], TEX1, 2D;
MULH H0, H0, H1; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
