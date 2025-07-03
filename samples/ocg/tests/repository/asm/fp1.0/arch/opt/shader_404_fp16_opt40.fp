!!FP2.0
TEX H1, f[TEX0], TEX0, 2D;
MULH H1.xyz, H1, f[COL0];
TEX H0, f[TEX2], TEX1, 2D;
MULH H0.w, H1, f[COL0];
MULH H0.xyz, H0, H1;
MADH H0.xyz, H0, {1.987654, 0, 0, 0}.x, {1.987654, 0, 0, 0}.y;
END

# Passes = 3 

# Registers = 1 

# Textures = 2 
