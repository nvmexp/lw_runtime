!!FP1.0 
TEX H0, f[TEX0], TEX0, 2D;
DP3H H1.x, f[TEX1], H0;
DP3H H1.y, f[TEX2], H0;
TEX H1, H1, TEX2, 2D;
END

# Passes = 4 

# Registers = 1 

# Textures = 3 
