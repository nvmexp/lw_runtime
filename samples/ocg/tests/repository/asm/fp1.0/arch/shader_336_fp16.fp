!!FP1.0
DECLARE C0={0.5, 0.5, 0.5, 0.5};
TEX H3, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
ADDH_SAT H0.w, {0, 0, 0, 1}, -H2.w;
MADH H0, H0.w, -H3, H3;
MADH H0, H0.w, H1, H0;
MULH H0, H0, H2;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH H0.w, H0.w;
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 2 

# Textures = 3 
