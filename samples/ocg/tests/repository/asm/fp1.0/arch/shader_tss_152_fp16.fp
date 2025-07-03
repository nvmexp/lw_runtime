!!FP1.0 
# Pixelshader 152
# Fog: Enabled as Vertex shader fog
# TSS count 5
MOVH H2, { 1.0, 0.0, 0.4, 0.7 }; # color & alpha paired
TXP H1, f[TEX1], TEX1, LWBE;
ADDH H2.xyz, H1, H2;
TXP H1, f[TEX2], TEX2, 2D;
ADDH H2.xyz, H1, H2;
MOVH H2.w, H1;
TXP H2.w, f[TEX3], TEX3, 2D; # eliminated a MOV
TXP H1, f[TEX4], TEX4, 2D;
MADH H0.xyz, H2.w, H1, H2;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
