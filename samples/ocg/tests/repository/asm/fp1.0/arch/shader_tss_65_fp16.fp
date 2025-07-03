!!FP1.0 
# Pixelshader 065
# TSS count 2
TXP H0, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
ADDH H3.xyz, H1, -H0;
MADH H0.xyz, H3, H1.w, H0;
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
