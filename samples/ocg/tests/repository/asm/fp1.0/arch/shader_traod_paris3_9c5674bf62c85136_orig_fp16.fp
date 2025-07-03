!!FP1.0
DECLARE C0={ 0.1, 0.3, 0.5, 0.9};
DECLARE C1={ 0.25, 0.20, 0.3, 0.23};
DECLARE C2={ 0.33, 0.13, 0.25, 0.20};
MOVH H1.xyz, f[TEX1];
MOVH H2, f[TEX0];
TEX H0, f[TEX2], TEX2, 2D;
DP3H H0.w, H0, H0;
LG2H H0.w, |H0.w|;
MULH H0.w, H0, C0.w;
EX2H H0.w, H0.w;
MULH H0.xyz, H0, H0.w;
MADH H0.xyz, H0, C0.y, H1;
MOVH H1.xyz, f[TEX3];
DP3H H0.w, H0, H0;
LG2H H0.w, |H0.w|;
MULH H0.w, H0, C0.w;
EX2H H0.w, H0.w;
MULH H3.xyz, H0, H0.w;
ADDH H0.xyz, H3, H3;
DP3H H0.w, H3, C1;
MADH H5.xyz, H0, H0.w, -C1;
DP3H H5.w, H1, H1;
MOVH H0.yzw, f[TEX3].wxyz;
LG2H H0.x, |H5.w|;
MULH H0.x, H0, C0.w;
EX2H H5.w, H0.x;
MULH H6.xyz, H5.w, H0.yzwy;
MADH H2.xy, H3, C1, H2;
MOVH H0, f[TEX0];
DP4H H0.x, H0, C1;
DP4H H4.w, H2, C1;
MOVH H1, f[TEX0];
DP4H H0.y, H1, C1;
MOVH H1, f[TEX0];
DP4H H0.w, H1, C2;
TXP H1, H0, TEX0, 2D;
DP4H H4.x, H2, C1;
DP4H H4.y, H2, C2;
ADDH H2.xyw, H4, -H0;
MOVH H4.xy, f[TEX0];
MULH H2.xyw, H1.w, H2;
DP3H_SAT H1.w, H5, H6;
LG2H H1.w, |H1.w|;
MULH H1.w, H1, C0.w;
EX2H H1.w, H1.w;
ADDH H2.xyw, H0, H2;
TXP H2, H2, TEX0, 2D;
ADDH H2.xyz, H2, -H1;
MADH H1.xyz, H2, H2.w, H1;
ADDH H1.xyz, H1.w, H1;
MULH H5.xyz, H1, C1;
TXP H2, H0, TEX1, 2D;
MOVH H1.zw, f[TEX0];
MADH H1.xy, H3, C0.y, H4;
DP4H H4.x, H1, C2;
DP4H H4.w, H1, C1;
DP4H H4.y, H1, C2;
ADDH H1.xyw, -H0, H4;
MADH H0.xyw, H1, H2.w, H0;
DP3H H2.w, H3, H6;
MADH_SAT H2.w, -H2, C0.y, C0.z;
TXP H0, H0, TEX2, 2D;
ADDH H0.xyz, H0, -H2;
MADH H2.xyz, H0, H0.w, H2;
MOVH H0.w, C0.z;
MADH H2.xyz, H2, C0.w, -H5;
MADH H0.xyz, H2, H2.w, H5;
MOVH o[COLH], H0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
