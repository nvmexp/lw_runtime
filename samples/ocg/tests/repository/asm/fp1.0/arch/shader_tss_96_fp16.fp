!!FP1.0 
# Pixelshader 096
# Fog: Enabled as Vertex shader fog
# TSS count 2
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, LWBE;
ADDH H3.xyz, H2, -{ 1.0, 0.0, 0.4, 0.7 };
MADH H0.xyz, H3, H1.w, { 1.0, 0.0, 0.4, 0.7 };
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
