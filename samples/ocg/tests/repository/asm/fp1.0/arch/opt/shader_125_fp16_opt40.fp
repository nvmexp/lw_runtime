!!FP2.0 
TEX H0, f[TEX2], TEX2, 2D;
DP3H_SAT H1.x, H0, f[TEX4];
TEX H0, f[TEX3], TEX3, 2D;
MULH H1, H1.x, H0;
DP3H_SAT H0.x, f[TEX1], f[TEX1];
ADDH H0.x, {1, 1, 1, 1}, -H0.x;
MULH H0, H0.x, H1;
MULH_m2 H0, H0, f[COL0];
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
