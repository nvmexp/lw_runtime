!!FP1.0 
# Pixelshader 299
# Fog: Enabled as Linear vertex fog
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H2, H1, f[COL0]; # color & alpha paired
MULH H0, { 1.0, 0.0, 0.4, 0.7 }, H2; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 1 
