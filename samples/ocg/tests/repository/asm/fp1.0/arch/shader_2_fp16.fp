!!FP1.0
TEXC HC, f[TEX0], TEX0, 2D;
KIL LT.xyzz;
TEX H1, f[TEX1], TEX1, 2D;
DP3H_SAT H2, f[TEX2], f[TEX2];
MOVH H0.xyz, H1;
ADDH H2, {1, 1, 1, 1}, -H2;
MULH H0.w, H1, H2;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 3 
