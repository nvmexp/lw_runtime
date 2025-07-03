!!FP2.0
TEX H1, f[TEX0], TEX0, 2D;
MULH H1, H1, f[COL0];
TEX H0, f[TEX2], TEX1, 2D;
MULH H1.xyz, H0, H1;
MULH H0, H1, {0.4, 0.4, 0.4, 1};
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
