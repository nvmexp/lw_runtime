!!FP1.0 
# Pixelshader 044
# Fog: Enabled as Linear vertex fog
# TSS count 1
TXP H1, f[TEX0], TEX0, 2D;
MULH H0, H1, { 1.0, 0.0, 0.4, 0.7 }; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 2 

# Textures = 1 
