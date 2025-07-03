!!FP1.0 
# Pixelshader 144
# Fog: Enabled as Vertex shader fog
# TSS count 3
TXP H1, f[TEX0], TEX0, 2D;
MOVH H2.xyz, H1;
ADDH_SAT H2.w, 1, -H1;
TXP H1, f[TEX1], TEX1, 2D;
ADDH H3.xyz, H1, -H2;
MADH H2.xyz, H3, H2.w, H2;
TXP H0.w, f[TEX2], TEX2, 2D; # eliminated a MOV
MOVR H0.xyz, H2;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 3 
