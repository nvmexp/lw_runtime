!!FP2.0
DECLARE C0= {0.816497, 0.000000, 0.577350, 0.000000};
DECLARE C1= {-0.408248, 0.707107, 0.577350, 0.000000};
DECLARE C2= {-0.408248, -0.707107, 0.577350, 0.000000};
DECLARE C3={2.000000, -1.000000, 0.000000, 0.000000};
TEX H0, f[TEX1], TEX4, 2D;
MADH H0.xyz, C3.x, H0, C3.y;
DP3H_SAT H2.y, H0, C1;
DP3H_SAT H2.x, H0, C0;
DP3H_SAT H2.w, H0, C2;
TEX H0, f[TEX2].zwww, TEX1, 2D;
MULH H0.xyz, H0, H2.x;
TEX H1, f[TEX2], TEX1, 2D;
MADH H2.xyz, H2.y, H1, H0;
TEX H0, f[TEX3].zwww, TEX1, 2D;
MADH H0.xyz, H2.w, H0, H2;
TEX H1, f[TEX0], TEX0, 2D;
MULH H1.xyz, H1, f[COL0];
MULH H1.xyz, H0, H1;
MULH H0.xyz, H1, C0.x;
MULH H0.w, H1, f[COL0];
END

# Passes = 10 

# Registers = 2 

# Textures = 4 
