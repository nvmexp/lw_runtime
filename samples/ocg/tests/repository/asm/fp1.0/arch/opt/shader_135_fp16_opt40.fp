!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={0.25, 0.2, 1, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX H0, f[TEX0], TEX3, 2D;
MADH H0, H0, C1.x, C1.y;
TEX H1, f[TEX3], TEX3, 2D;
MADH H1, H1, C1.x, C1.y;
MOVH H3, f[TEX1];
ADDH H0, H0, H1;
MULH H0, H0, C2.x;
DP3H H1.x, H0, f[TEX2];
ADDH H1.y, H1.x, H1.x;
MADH H0, H1.y, H0, -f[TEX2];
TEX H0, H0, TEX7, LWBE;
ADDH H1.x, C1.w, -H1;
LG2H H1.x, H1.x;
MULH H1.x, C1.z, H1.x;
EX2H H1.x, H1.x;
TEX H2, f[TEX0], TEX0, 2D;
MADH H2, H2, C1.x, C1.y;
DIVH H3, H3, H3.w;
MADH H2, H2, C0, H3;
TEX H3, H2, TEX1, 2D;
TEX H2, H2, TEX2, 2D;
ADDH H0, H0, H3;
TEX H3, f[TEX4], TEX5, 2D;
MULH H1.x, H1.x, H3.w;
MADH H2, H1.x, -H2, H2;
MADH H0, H1.x, H0, H2;
MULH H0, H0, H3;
END

# Passes = 18 

# Registers = 2 

# Textures = 5 
