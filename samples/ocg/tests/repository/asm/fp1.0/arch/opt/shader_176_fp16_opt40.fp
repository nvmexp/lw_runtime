!!FP2.0 
TEX H0, f[TEX0], TEX0, 2D;
DP3H H1.x, f[TEX1], H0;
DP3H H1.y, f[TEX2], H0;
DP3H H1.z, f[TEX3], H0;
TEX H1, H1, TEX6, 3D;
MULH H0, H1, f[COL0];
END

# Passes = 6 

# Registers = 1 

# Textures = 4 
