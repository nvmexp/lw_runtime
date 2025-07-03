!!FP1.0 
# Pixelshader 043
# Fog: Enabled as Linear vertex fog
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H2, H1, { 1.0, 0.0, 0.4, 0.7 }; # color & alpha paired
TXP H1, f[TEX1], TEX1, 2D;
MULH H0, H1, H2; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
