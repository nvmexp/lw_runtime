!!FP1.0 
# Pixelshader 270
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MOVH H0.xyz, H1;
ADDH H0.w, H1, -f[COL0];
TXP H1, f[TEX1], TEX1, 2D;
ADDH H3.xyz, { 1.0, 0.0, 0.4, 0.7 }, -H1;
MADH H0.xyz, H3, H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
