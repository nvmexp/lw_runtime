!!FP1.0 
# Pixelshader 188
# Fog: Enabled as Vertex shader fog
# TSS count 5
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
MULH H2.xyz, H1, H2;
MULH H2.xyz, f[COL0], H2;
MULH H2.xyz, H2, {2, 0, 0, 0}.x; 
TXP H2.w, f[TEX3], TEX3, 2D; # eliminated a MOV
TXP H1, f[TEX4], TEX4, LWBE;
MADH H0.xyz, H2.w, H1, H2;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
