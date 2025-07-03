!!FP1.0 
# Pixelshader 139
# Fog: Enabled as Vertex shader fog
# TSS count 5
TXP H2, f[TEX0], TEX0, LWBE; # eliminated a MOV
TXP H2.w, f[TEX1], TEX1, 2D; # eliminated a MOV
TXP H1, f[TEX2], TEX2, 2D;
ADDH H3.xyz, H1, -H2;
MADH H2.xyz, H3, H2.w, H2;
MULH H2.xyz, f[COL0], H2;
MULH H2.xyz, H2, {2, 0, 0, 0}.x; 
TXP H0.w, f[TEX4], TEX4, 2D; # eliminated a MOV
MOVR H0.xyz, H2;
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
