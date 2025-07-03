!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.500000, 0.500000, 0.250000, 0.000000};
DECLARE C3={1, 2, 3, 4};
DECLARE C4={4.000000, -1.000000, 0.000000, 0.500000};
DECLARE C5={2.000000, 0.450000, 0.250000, 1.000000};
DECLARE C6={1.500000, 0.000000, 0.000000, 0.000000};
DECLARE C7={2.000000, -1.000000, 0.000000, 0.000000};
MULH H0.xyz, f[TEX0], C2.z;
MULH H7.xyz, H0, C4.x;
MULH H2.xyz, f[TEX0].y, C3.y;
TEX H9, H0, TEX6, 3D;
TEX H4, H7, TEX6, 3D;
TEX H11, H2, TEX6, 3D;
MADH H11.w, H4.x, C4.w, H9.x;
MADH H0.xyz, H11.w, C2.y, f[TEX0];
MADH H0.w, C7.x, H11.x, C7.y;
MULH H1.xz, H0.w, C3.x;
MULH H1.y, H0.w, C4.z;
ADDH H8.xyz, H0, H1;
MULH H3.xy, H8, C3.w;
MULH H3.z, H8.y, C4.z;
TEX H10, H3, TEX6, 3D;
TEX H5, f[TEX1], TEX1, 2D;
MULH H5.w, H8.x, H8.x;
MADH H5.w, H8.z, H8.z, H5.w;
RSQH H7.w, H5.w;
MULH H5.w, H7.w, H5.w;
MULH H5.w, H5.w, C2.x;
MULH H10.w, H5.w, C4.w;
ADDH H10.w, H10.w, C5.x;
RCPH H0.w, H5.w;
MULH_SAT H10.w, H10.w, H0.w;
MULH H10.w, H10.w, C3.z;
MADH H4.w, C7.x, H10.x, C7.y;
MADH H5.w, H4.w, H10.w, H5.w;
FRCH H5.w, H5.w;
MULH H11.w, H5.w, C4.w;
RCPH H5.w, H5.w;
ADDH H0.w, H11.w, C5.z;
ADDH H1.w, H11.w, C5.y;
MULH H8.w, H0.w, H5.w;
MULH H5.w, H1.w, H5.w;
MINH H0.w, H8.w, C5.w;
ADDH H10.w, H5.w, C4.y;
MOVHC HC, H10.w;
MOVH H5.w(GE), C5.w;
MOVH H5.w(LT), H5.x;
ADDH H5.w, -H0.w, H5.w;
MOVH H2.xyz, -C0;
ADDH H4.xyz, H2, C1;
MADH H11.xyz, H4, H5.w, C0;
MULH H0.xyz, H5, H11;
DP4H H0.w, f[TEX4], f[TEX4];
RSQH H0.w, H0.w;
MULH H1.xyz, H0.w, f[TEX4];
DP4H H1.w, f[TEX3], f[TEX3];
RSQH H1.w, H1.w;
MULH H8.xyz, H1.w, f[TEX3];
DP3H H0.w, H1, H8;
MOVHC HC, H0.w;
MOVH H0.w(GE), H0.w;
MOVH H0.w(LT), C4.x;
MULH H0.w, H0.w, H0.w;
MULH H0.w, H0.w, C6.x;
ADDH H0.w, H0.w, C5.w;
MULH H0.xyz, H0, H0.w;
MOVH H0.w, C4.z;
MOVH o[COLH], H0; 
END

# Passes = 47 

# Registers = 6 

# Textures = 4 
