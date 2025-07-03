!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H0, f[TEX0], TEX0, 2D;
DP3H H1.x, f[TEX1], H0;
DP3H H1.y, f[TEX2], H0;
TEX H0, f[TEX3], TEX3, 2D;
MADH H2, f[COL0], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H_SAT H2, H2, H0;
ADDH H2.w, {1, 1, 1, 1}, -H2.w;
MULH H0.w, H2.w, H2.w;
MULH H0.w, H0.w, H0.w;
MULH H0.w, H0.w, H2.w;
ADDH H2.w, {1, 1, 1, 1}, -C0.w;
MADH H0.w, H0.w, H2.w, C0.w;
TEX H1, H1, TEX2, 2D;
MULH H0, H0.w, H1;
END

# Passes = 10 

# Registers = 2 

# Textures = 4 
