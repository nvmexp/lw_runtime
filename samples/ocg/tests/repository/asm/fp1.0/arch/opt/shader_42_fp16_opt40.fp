!!FP2.0 
DECLARE C0={0.9, 0.8, 0.7, 0.6};
DECLARE C1={0.4, 0.3, 0.2, 0.1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX H0, f[TEX0], TEX0, 2D;
DP3H H1.x, f[TEX1], H0;
DP3H H1.y, f[TEX2], H0;
DP3H H1.z, f[TEX3], H0;
DP3H_m2 H1.w, H1, f[15];
DP3H H0.w, f[15], f[15];
DIVH H1.w, H1.w, H0.w;
MADH H1.xyz, H1.w, f[15], -H1;
TEX H1, H1, TEX6, 3D;
MULH H0.xyz, H1, C0;
MOVH H0.w, C0.w;
MULH H1.xyz, H0, H0;
MADH H0.xyz, C1, -H0, H0;
MADH H0.xyz, C1, H1, H0;
DP3H H1.xyz, H0, C3;
MADH H1.xyz, C2, -H1, H1;
MADH H0.xyz, C2, H0, H1;
MULH H0.xyz, H0.w, H0;
END

# Passes = 15 

# Registers = 1 

# Textures = 4 
