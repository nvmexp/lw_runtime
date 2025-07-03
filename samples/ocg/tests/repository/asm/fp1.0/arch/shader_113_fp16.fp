!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
TEX H3, f[TEX3], TEX3, 2D;
MULH H0, H0, H1;
MULH H0, H0, f[COL0];
MULH H0, H0, H2;
MULH H0, H0, H3;
MULH H0, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
