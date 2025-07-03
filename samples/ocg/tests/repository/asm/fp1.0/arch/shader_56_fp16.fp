!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
MADH H5, f[COL1], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H H3, H0, H5;
ADDH H2, H3, C0;
MULH H3, f[COL0], H2;
MULH H0.xyz, H3, H1;
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 3 

# Textures = 2 
