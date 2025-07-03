!!FP1.0 
# Pixelshader 149
# Fog: Enabled as Vertex shader fog
# TSS count 3
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
ADDH H3.xyz, H1, -H2;
MADH H2.xyz, H3, f[COL0].w, H2;
ADDH H3.w, H1, -H2;
MADH H2.w, H3, f[COL0].w, H2;
MULH H0.xyz, f[COL0], H2;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
