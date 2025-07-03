!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.500000, 0.500000, 0.250000, 0.000000};
DECLARE C3={1, 2, 3, 4};
DECLARE C4={4.000000, -1.000000, 0.000000, 0.500000};
DECLARE C5={2.000000, 0.450000, 0.250000, 1.000000};
DECLARE C6={1.500000, 0.000000, 0.000000, 0.000000};
DECLARE C7={2.000000, -1.000000, 0.000000, 0.000000};
MOVH H3.xyz, f[TEX0];
MULH H0.xyz, H3, C2.z;
TEX H1.x, H0, TEX6, 3D;
MULH H2.xyz, H0, C4.x;
TEX H2.x, H2, TEX6, 3D;
MADH H2.w, H2.x, C4.w, H1.x;
MULH H0.xyz, H3.y, C3.y;
TEX H2.x, H0, TEX6, 3D;
MADH H0.xyz, H2.w, C2.y, H3;
MADH H0.w, C7.x, H2.x, C7.y;
MULH H1.xz, H0.w, C3.x;
MULH H1.y, H0.w, C4.z;
ADDH H0.xyz, H0, H1;
MULH H3.xy, H0, C3.w;
MULH H3.z, H0.y, C4.z;
TEX H1, H3, TEX6, 3D;
TEX H3, f[TEX1], TEX1, 2D;
MULH H3.w, H0.x, H0.x;
MADH H3.w, H0.z, H0.z, H3.w;
LG2H_d2 H2.w, H3.w;
EX2H H2.w, -H2.w;
MULH H3.w, H2.w, H3.w;
MULH H3.w, H3.w, C2.x;
MULH H1.w, H3.w, C4.w;
ADDH H1.w, H1.w, C5.x;
DIVH_SAT H1.w, H1.w, H3.w;
MULH H1.w, H1.w, C3.z;
MADH H2.w, C7.x, H1.x, C7.y;
MADH H3.w, H2.w, H1.w, H3.w;
FRCH H3.w, H3.w;
MULH H1.w, H3.w, C4.w;
ADDH H0.w, H1.w, C5.z;
DIVH H2.w, H0.w, H3.w;
ADDH H1.w, H1.w, C5.y;
DIVH H3.w, H1.w, H3.w;
MINH H0.w, H2.w, C5.w;
ADDH H1.w, H3.w, C4.y;
MOVH HC, H1.w;
MOVH H3.w(GE), C5.w;
MOVH H3.w(LT), H3.x;
ADDH H3.w, -H0.w, H3.w;
MOVH H0.xyz, -C0;
ADDH H2.xyz, H0, C1;
MOVH H1, f[TEX4];
MADH H2.xyz, H2, H3.w, C0;
MULH H0.xyz, H3, H2;
DP4H H0.w, H1, H1;
LG2H_d2 H0.w, H0.w;
MOVH H2, f[TEX3];
EX2H H0.w, -H0.w;
MULH H1.xyz, H0.w, H1;
DP4H H1.w, H2, H2;
LG2H_d2 H1.w, H1.w;
EX2H H1.w, -H1.w;
MULH H2.xyz, H1.w, H2;
DP3H H0.w, H1, H2;
MOVH HC, H0.w;
MOVH H0.w(GE), H0.w;
MOVH H0.w(LT), C4.x;
MULH H0.w, H0.w, H0.w;
MULH H0.w, H0.w, C6.x;
ADDH H0.w, H0.w, C5.w;
MADH H0.xyz, H0, H0.w, {0, 0, 0, 0}.x;
MOVH H0.w, C4.z;
END

# Passes = 38 

# Registers = 2 

# Textures = 4 
