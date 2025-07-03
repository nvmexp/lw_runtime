!!FP1.0 
# Pixelshader 147
# Fog: Enabled as Vertex shader fog
# TSS count 3
TXP H1, f[TEX0], TEX0, 2D;
MULH H2.xyz, H1, f[COL0];
MULH H2.xyz, H2, {2, 0, 0, 0}.x; 
MULH H2.w, H1, f[COL0];
TXP H2.w, f[TEX1], TEX1, 2D; # eliminated a MOV
TXP H1, f[TEX2], TEX2, 2D;
ADDH H3.xyz, H1, -H2;
MADH H0.xyz, H3, H2.w, H2;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 3 
