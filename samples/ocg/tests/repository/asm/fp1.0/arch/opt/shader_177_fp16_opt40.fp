!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
DP3H H0.xyz, H0, H1;
MOVH H0.w, H1;
MULH H0, H0, C0;
ADDH_SAT H0, H0, C1;
MULH H0.xyz, H0, f[COL0];
ADDH_m4_SAT H0.w, H0, H0;
MULH H0.xyz, H0, H0.w;
END

# Passes = 5 

# Registers = 1 

# Textures = 2 
