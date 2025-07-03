!!FP1.0
TEX H4, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
TEX H3, f[TEX3], TEX3, 2D;
ADDH H0, f[COL0], H3;
MULH H0, H4, H0;
MADH H0, H1, H2, H0;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 3 

# Textures = 4 
