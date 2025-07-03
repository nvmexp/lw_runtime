!!FP1.0 
# Pixelshader 191
# Fog: Enabled as Vertex shader fog
# TSS count 2
MULH H2.xyz, { 1.0, 0.0, 0.4, 0.7 }, f[COL0];
MULH H2.xyz, H2, {2, 0, 0, 0}.x; 
MULH H2.w, { 1.0, 0.0, 0.4, 0.7 }, f[COL0];
TXP H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H1, H2;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 1 
