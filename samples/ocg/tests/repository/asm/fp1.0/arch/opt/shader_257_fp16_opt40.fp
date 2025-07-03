!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX3], TEX3, 2D;
DP3H_SAT H0.xyz, H0, H1;
TEX H1, f[TEX2], TEX2, 2D;
MOVH H0.w, H1;
MULH H0.xyz, H0, H0;
ADDH H0.w, H0, C0;
MULH_SAT H0.w, H0, C1;
MULH H0.xyz, H0, H0;
ADDH_m4_SAT H0.w, H0, H0;
TEX H1, f[TEX0], TEX0, 2D;
MULH H1.xyz, H0.w, H1;
MULH H1.xyz, f[COL0], H1;
MULH H0.xyz, H0, H0;
MULH H0.xyz, H0, H1;
END

# Passes = 8 

# Registers = 1 

# Textures = 3 
