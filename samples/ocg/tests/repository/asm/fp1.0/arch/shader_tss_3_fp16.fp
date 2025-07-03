!!FP1.0 
# Pixelshader 003
# TSS count 1
TXP H2, f[TEX0], TEX0, 2D;
MULH H0, f[COL0], H2; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = 1 

# Registers = 3 

# Textures = 1 
