!!FP1.0 
# Pixelshader 213
# Fog: Enabled as Linear vertex fog
# TSS count 3
TXP H1, f[TEX0], TEX0, 2D;
MULH H2.xyz, H1, f[COL0];
MOVH H2.w, H1;
TXP H1, f[TEX1], TEX1, 2D;
MULH H2.xyz, H1, H2;
TXP H1, f[TEX2], TEX2, 2D;
MULH H0.xyz, H1, H2;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 3 
