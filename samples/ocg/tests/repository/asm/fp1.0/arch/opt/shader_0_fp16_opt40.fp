!!FP2.0
TEX H1, f[TEX1], TEX1, 2D;
DP3H H2.x, f[TEX2], H1;
TEX H0, f[TEX0], TEX0, 2D;
DP3H H2.y, f[TEX3], H1;
TEX H2, H2, TEX3, 2D;
MULH H2, H0, H2;
MULH_m2 H0, H2, f[COL0];
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
