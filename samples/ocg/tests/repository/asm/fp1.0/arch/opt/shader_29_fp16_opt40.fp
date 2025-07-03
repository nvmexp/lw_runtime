!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX H0, f[TEX0], TEX0, 2D;
DP3H_SAT H3.x, H0, C0;
TEX H1, f[TEX1], TEX1, 2D;
MULH H1, H1, H3.x;
DP3H_SAT H3.x, H0, C1;
TEX H2, f[TEX2], TEX2, 2D;
MADH H1, H3.x, H2, H1;
TEX H2, f[TEX3], TEX3, 2D;
DP3H_SAT H3.x, H0, C2;
MADH H1, H3.x, H2, H1;
MADH H0, H1, C3, H3.w;
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
