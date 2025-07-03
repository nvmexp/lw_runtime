!!FP1.0 
# Pixelshader 080
# Fog: Enabled as Vertex shader fog
# TSS count 3
TXP H2, f[TEX0], TEX0, LWBE; # eliminated a MOV
TXP H2.w, f[TEX1], TEX1, 2D; # eliminated a MOV
TXP H1, f[TEX2], TEX2, LWBE;
MADH H0.xyz, H2.w, H1, H2;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
