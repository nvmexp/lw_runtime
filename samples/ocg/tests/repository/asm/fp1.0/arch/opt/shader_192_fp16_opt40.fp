!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.500000, 0.500000, 0.250000, 0.000000};
DECLARE C3={6.283190, -0.000000, 0.000025, -3.141590};
DECLARE C4={-0.001389, -0.500000, 1.000000, 0.041667};
DECLARE C5={0.000000, 1.200000, 0.400000, 20.000000};
DECLARE C6={2.000000, -1.000000, 0.000000, 0.000000};
MADH H0.xyz, f[TEX0], C2.x, C2.x;
TEX H1, H0, TEX6, 3D;
TEX H0, f[TEX1], TEX1, 2D;
MADH H0.w, C6.x, H1.x, C6.y;
MADH H0.w, f[TEX0].x, C0.w, H0.w;
MADH H0.w, H0.w, C2.y, C2.z;
FRCH H0.w, H0.w;
MADH H0.w, H0.w, C3.x, C3.w;
MULH H0.w, H0.w, H0.w;
MADH H1.w, H0.w, C3.y, C3.z;
MADH H1.w, H0.w, H1.w, C4.x;
MADH H1.w, H0.w, H1.w, C4.w;
MADH H1.w, H0.w, H1.w, C4.y;
MADH H0.w, H0.w, H1.w, C4.z;
LG2H H0.w, |H0.w|;
MULH H0.w, H0.w, C2.x;
EX2H H0.w, H0.w;
MOVH H1.xyz, -C1;
ADDH H1.xyz, H1, C0;
MADH H1.xyz, H1, H0.w, C1;
MULH H1.xyz, H0, H1;
MULH H0.xyz, H1, C5.y;
DP4H H0.w, f[TEX3], f[TEX3];
LG2H_d2 H0.w, H0.w;
EX2H H0.w, -H0.w;
MULH H1.xyz, H0.w, f[TEX3];
DP3H H0.w, f[TEX4], H1;
MOVH HC, H0.w;
MOVH H0.w(LT), C5.x;
LG2H H0.w, H0.w;
MULH H0.w, H0.w, C5.w;
EX2H H0.w, H0.w;
MULH H0.w, H0.w, C5.z;
ADDH H0.w, H0.w, C4.z;
MULH H0.xyz, H0, H0.w;
MOVH H0.w, C4.z;
END

# Passes = 27 

# Registers = 1 

# Textures = 4 
