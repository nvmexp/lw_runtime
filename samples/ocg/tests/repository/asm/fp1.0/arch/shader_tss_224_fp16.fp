!!FP1.0 
# Pixelshader 224
# Fog: Enabled as Linear vertex fog
# TSS count 3
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
MULH H2.xyz, H1, H2;
TXP H1, f[TEX2], TEX2, 2D;
ADDH H0.xyz, H1, H2;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
