!!FP2.0
DECLARE C0={0.1, 0.1, 0.1, 0.1};
DECLARE C1={0.7, 0.71, 0.71, 0.71};
DECLARE C2={0.5, 0.5, 0.5, 0.5};
DECLARE C3={0.3, 0.3, 0.3, 0.3};
DECLARE C4={0.4, 0.4, 0.4, 0.4};
DECLARE C5={0.816497, 0.000000, 0.577350, 0.000000};
DECLARE C6={-0.408248, 0.707107, 0.577350, 0.000000};
DECLARE C7={-0.408248, -0.707107, 0.577350, 0.000000};
DECLARE C8={2.000000, -1.000000, 0.333333, 0.000000};
TEX H2, f[TEX1], TEX3, 2D;
MADH H2.xyz, C8.x, H2, C8.y;
MULH H0.xyz, H2.y, f[TEX5];
MADH H0.xyz, f[TEX4], H2.x, H0;
MADH H0.xyz, f[TEX6], H2.z, H0;
DP3H H1.w, H0, H0;
RCPH H0.w, H1.w;
DP3H H1.x, H0, f[TEX3];
MULH_m2 H0.w, H0, H1.x;
MADH H0.xyz, H0.w, H0, -f[TEX3];
TEX H1, H0, TEX1, 2D;
MULH H1.xyz, H2.w, H1;
MULH H3.xyz, H1, C0;
DP3H_SAT H1.x, H2, C6;
TEX H0, f[TEX0], TEX0, 2D;
MULH H1.xyz, H1.x, f[COL1];
DP3H_SAT H3.w, H2, C5;
DP3H_SAT H2.w, H2, C7;
MADH H1.xyz, H3.w, f[COL0], H1;
MADH H2.xyz, H3, H3, -H3;
MADH H1.xyz, H2.w, f[TEX7], H1;
MADH H2.xyz, C2, H2, H3;
MULH H1.xyz, H1, C1;
MULH H0.xyz, H0, H1;
DP3H H3.x, H2, C8.z;
MADH H3.xyz, C3, -H3.x, H3.x;
MULH H0.xyz, H0, C4;
MADH H1.xyz, C3, H2, H3;
MULH H0.w, H0.w, C1.w;
MADH H0.xyz, C8.x, H0, H1;
END

# Passes = 22 

# Registers = 2 

# Textures = 6 
