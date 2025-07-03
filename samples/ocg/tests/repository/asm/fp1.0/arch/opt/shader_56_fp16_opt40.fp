!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H0, f[TEX0], TEX0, 2D;
MADH H1, f[COL1], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H H1, H0, H1;
ADDH H1, H1, C0;
MULH H0, f[COL0], H1;
TEX H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H0, H1;
MOVH H0.w, H1;
END

# Passes = 5 

# Registers = 1 

# Textures = 2 
