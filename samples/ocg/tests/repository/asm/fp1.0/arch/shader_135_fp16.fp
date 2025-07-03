!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={0.25, 0.2, 1, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX H0, f[TEX0], TEX3, 2D;
TEX H1, f[TEX3], TEX3, 2D;
TEX H2, f[TEX0], TEX0, 2D;
MADH H2, H2, C1.x, C1.y;
RCPH H3.w, f[TEX1].w;
MULH H3, H3.w, f[TEX1];
MADH H3, H2, C0, H3;
MADH H0, H0, C1.x, C1.y;
MADH H1, H1, C1.x, C1.y;
ADDH H0, H0, H1;
MULH H0, H0, C2.x;
DP3H H1.x, H0, f[TEX2];
ADDH H2.x, H1.x, H1.x;
MADH H0, H2.x, H0, -f[TEX2];
ADDH H4.x, C1.w, -H1;
POWH H4.x, H4.x, C1.z;
TEX H0, H0, TEX7, LWBE;
TEX H1, H3, TEX1, 2D;
TEX H2, H3, TEX2, 2D;
TEX H3, f[TEX4], TEX5, 2D;
MULH H4.x, H4.x, H3.w;
ADDH H0, H0, H1;
MADH H5, H4.x, -H2, H2;
MADH H0, H4.x, H0, H5;
MULH H0, H0, H3;
MOVH o[COLH], H0; 
END

# Passes = 25 

# Registers = 3 

# Textures = 5 
