!!FP1.0 
# Pixelshader 151
# Fog: Enabled as Vertex shader fog
# TSS count 4
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H2.w, f[TEX1], TEX1, 2D; # eliminated a MOV
TXP H1, f[TEX2], TEX2, 2D;
ADDH H3.xyz, H1, -H2;
MADH H2.xyz, H3, H2.w, H2;
TXP H1, f[TEX3], TEX3, 2D;
MULH H0.xyz, H1, H2;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
