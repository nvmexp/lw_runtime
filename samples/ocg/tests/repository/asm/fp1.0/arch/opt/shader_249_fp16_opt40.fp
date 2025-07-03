!!FP2.0
TEX H0, f[TEX3], TEX3, 2D;
ADDH H0, f[COL0], H0;
TEX H1, f[TEX0], TEX0, 2D;
MADH H0, H1, H0, f[COL1];
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
MADH H0, H1, H2, H0;
END

# Passes = 4 

# Registers = 2 

# Textures = 4 
