!!FP1.0 
# Pixelshader 101
# Fog: Enabled as Vertex shader fog
# TSS count 3
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
MULH H2.xyz, H1, H2;
# alpha disabled
MULH H0.xyz, f[COL0], H2;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
