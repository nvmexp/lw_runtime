!!FP2.0
DECLARE C0= {0.21, 0.36, 0.54, 0.45};
DECLARE C1= {0.54, 0.65, 0.54, 0.34};
DECLARE C2= {0.6, 0.5, 0.7, 0.9};
DECLARE C3= {0.3, 0.2, 0.5, 0.6};
TEX H0, f[TEX0], TEX0, 2D;
DP3H_SAT H1, H0, C0;
TEX H0, f[TEX1], TEX1, 2D;
MULH H0, H0, H1;
TEX H2, f[TEX2], TEX2, 2D;
DP3H_SAT H1, H0, C1;
MADH H0, H1, H2, H0;
TEX H2, f[TEX3], TEX3, 2D;
DP3H_SAT H1, H0, C2;
MADH H0, H1, H2, H0;
MADH H0, H0, C3, H31;
END

# Passes = 7 

# Registers = 16 

# Textures = 4 
