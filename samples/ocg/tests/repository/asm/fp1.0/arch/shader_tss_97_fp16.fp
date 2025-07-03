!!FP1.0 
# Pixelshader 097
# Fog: Enabled as Vertex shader fog
# TSS count 3
MOVH H2, { 1.0, 0.0, 0.4, 0.7 }; # color & alpha paired
TXP H1, f[TEX1], TEX1, LWBE;
ADDH H2.xyz, H1, H2;
# alpha disabled
TXP H1, f[TEX2], TEX2, 2D;
ADDH H0.xyz, H1, H2;
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
