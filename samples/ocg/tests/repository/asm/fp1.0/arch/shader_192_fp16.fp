!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.500000, 0.500000, 0.250000, 0.000000};
DECLARE C3={6.283190, -0.000000, 0.000025, -3.141590};
DECLARE C4={-0.001389, -0.500000, 1.000000, 0.041667};
DECLARE C5={0.000000, 1.200000, 0.400000, 20.000000};
DECLARE C6={2.000000, -1.000000, 0.000000, 0.000000};
MADH H7.xyz, f[TEX0], C2.x, C2.x;
TEX H2, H7, TEX6, 3D;
TEX H9, f[TEX1], TEX1, 2D;
MADH H9.w, C6.x, H2.x, C6.y;
MADH H9.w, f[TEX0].x, C0.w, H9.w;
MADH H9.w, H9.w, C2.y, C2.z;
FRCH H9.w, H9.w;
MADH H9.w, H9.w, C3.x, C3.w;
MULH H9.w, H9.w, H9.w;
MADH H6.w, H9.w, C3.y, C3.z;
MADH H8.w, H9.w, H6.w, C4.x;
MADH H10.w, H9.w, H8.w, C4.w;
MADH H0.w, H9.w, H10.w, C4.y;
MADH H9.w, H9.w, H0.w, C4.z;
LG2H H9.w, |H9.w|;
MULH H9.w, H9.w, C2.x;
EX2H H9.w, H9.w;
MOVH H2.xyz, -C1;
ADDH H4.xyz, H2, C0;
MADH H6.xyz, H4, H9.w, C1;
MULH H1.xyz, H9, H6;
MULH H8.xyz, H1, C5.y;
DP4H H8.w, f[TEX3], f[TEX3];
RSQH H8.w, H8.w;
MULH H3.xyz, H8.w, f[TEX3];
DP3H H8.w, f[TEX4], H3;
MOVHC HC, H8.w;
MOVH H8.w(GE), H8.w;
MOVH H8.w(LT), C5.x;
LG2H H8.w, H8.w;
MULH H8.w, H8.w, C5.w;
EX2H H8.w, H8.w;
MULH H8.w, H8.w, C5.z;
ADDH H8.w, H8.w, C4.z;
MULH H0.xyz, H8, H8.w;
MOVH H0.w, C4.z;
MOVH o[COLH], H0; 
END

# Passes = 28 

# Registers = 6 

# Textures = 4 
