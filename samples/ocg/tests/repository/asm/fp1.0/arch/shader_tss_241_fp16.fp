!!FP1.0 
# Pixelshader 241
# TSS count 3
TXP H0, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
ADDH H3.xyz, H1, -H0;
MADH H0.xyz, H3, H0.w, H0;
# alpha disabled
TXP H1, f[TEX2], TEX2, 2D;
ADDH H3.xyz, H0, -H1;
MADH H0.xyz, H3, H0.w, H1;
# alpha disabled
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 3 
