!!FP1.0
DECLARE diffuse = {1,1,1,1};
DECLARE spelwlar = {1,1,1,1};
TEX      H0, f[TEX1], TEX1, 2D;
MADX     H0, H0, 2, -1;
TEX      H1, f[TEX0], TEX0, LWBE;
MADX     H1, H1, 2, -1;
DP3X_SAT H1, H0, H1;
TEX      H2, f[TEX6], TEX0, LWBE;
MADX     H2, H2, 2, -1;
TEX      H3, f[TEX2], TEX2, 2D;
DP3X_SAT H0, H0, H2;
TXP      H2, f[TEX3], TEX3, 2D;
TEX      H0, H0, TEX6, 2D;
MULX     H3, H3, H2;
MULX     H1, H3, H1;
TEX      H2, f[TEX4], TEX4, 2D;
TEX      H3, f[TEX5], TEX5, 2D;
MULX     H2, H2, diffuse;
MULH     H3, H3, spelwlar;
MADX     H0, H3, H0, H2;
MULX     o[COLH], H1, H0;
END
