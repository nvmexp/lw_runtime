!!FP1.0 
# Pixelshader 247
# Fog: Enabled as Vertex shader fog
# TSS count 4
MOVH H2, { 1.0, 0.0, 0.4, 0.7 }; # color & alpha paired
TXP H1, f[TEX1], TEX1, 2D;
MULH H2.xyz, H1, H2;
MULH H2.xyz, f[COL0], H2;
MULH H2.xyz, H2, {2, 0, 0, 0}.x; 
TXP H0.w, f[TEX3], TEX3, 2D; # eliminated a MOV
MOVR H0.xyz, H2;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
