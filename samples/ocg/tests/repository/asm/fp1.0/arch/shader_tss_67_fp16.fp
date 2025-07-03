!!FP1.0 
# Pixelshader 067
# Fog: Enabled as Vertex shader fog
# TSS count 4
TXP H1, f[TEX0], TEX0, 2D;
MULH H0.xyz, H1, f[COL0];
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
# alpha disabled
TXP H1, f[TEX1], TEX1, 2D;
MULH H2.xyz, H0, H1;
# alpha disabled
TXP H1, f[TEX2], TEX2, 2D;
MULH H0.xyz, H1, f[COL0];
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
# alpha disabled
TXP H1, f[TEX3], TEX3, 2D;
MADH H0.xyz, H2, H0, H1;
# alpha disabled
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
