!!FP1.0 
# Pixelshader 274
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H0, H1, f[COL0]; # color & alpha paired
MULH H0, { 1.0, 0.0, 0.4, 0.7 }, H0; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
