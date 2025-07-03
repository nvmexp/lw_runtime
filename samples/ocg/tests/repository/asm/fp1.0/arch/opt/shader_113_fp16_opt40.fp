!!FP2.0
TEX H0, f[TEX0], TEX0, 2D;
MULH H0, H0, f[COL0];
TEX H1, f[TEX1], TEX1, 2D;
MULH H0, H0, H1;
TEX H1, f[TEX2], TEX2, 2D;
MULH H0, H0, H1;
TEX H1, f[TEX3], TEX3, 2D;
MULH_m2 H0, H0, H1;
END

# Passes = 4 

# Registers = 1 

# Textures = 4 
