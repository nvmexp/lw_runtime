!!FP2.0
TEX HC, f[TEX0], TEX0, 2D;
KIL LT.xyzz;
DP3H_SAT H1, f[TEX2], f[TEX2];
ADDH H1, {1, 1, 1, 1}, -H1;
TEX H0, f[TEX1], TEX1, 2D;
MULH H0.w, H0, H1;
END

# Passes = 4 

# Registers = 1 

# Textures = 3 
