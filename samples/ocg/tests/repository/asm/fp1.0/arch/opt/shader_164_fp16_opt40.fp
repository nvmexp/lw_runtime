!!FP2.0 
TEX H1, f[TEX5], TEX5, 2D;
TEX H0, f[TEX5], TEX5, 2D;
DP3H_SAT H1.xyz, H1, H0;
TEX H0, f[TEX2], TEX2, 2D;
MOVH_SAT H1.w, H0;
MULH H1.xyz, H1, H1;
ADDH_m4_SAT H1.w, H1, H1;
TEX H0, f[TEX0], TEX0, 2D;
MULH H1.xyz, H1, H1;
MULH H0.xyz, H1.w, H0;
MULH H1.xyz, H1, H1;
MULH H0.xyz, f[COL0], H0;
MULH H1.xyz, H1, H1;
MULH H0.xyz, H1, H0;
END

# Passes = 8 

# Registers = 1 

# Textures = 3 
