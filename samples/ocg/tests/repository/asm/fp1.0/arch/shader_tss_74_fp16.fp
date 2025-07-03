!!FP1.0 
# Pixelshader 074
# Fog: Enabled as Vertex shader fog
# TSS count 1
TXP H1, f[TEX0], TEX0, 2D;
MULH H0, H1, f[COL0]; # color & alpha paired
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 2 

# Textures = 1 
