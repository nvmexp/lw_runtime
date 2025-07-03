!!FP1.0 
# Pixelshader 098
# Fog: Enabled as Vertex shader fog
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
ADDH H3.xyz, H1, -{ 1.0, 0.0, 0.4, 0.7 };
MADH H2.xyz, H3, f[COL0].w, { 1.0, 0.0, 0.4, 0.7 };
MOVH H2.w, H1;
TXP H1, f[TEX1], TEX1, 2D;
ADDH H3.xyz, H2, -{ 1.0, 0.0, 0.4, 0.7 };
MADH H0.xyz, H3, H1.w, { 1.0, 0.0, 0.4, 0.7 };
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
