!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={0.25, 0.2, 1, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R0, f[TEX0], TEX3, 2D;
TEX R1, f[TEX3], TEX3, 2D;
TEX R2, f[TEX0], TEX0, 2D;
MADR R2, R2, C1.x, C1.y;
RCPR R3.w, f[TEX1].w;
MULR R3, R3.w, f[TEX1];
MADR R3, R2, C0, R3;
MADR R0, R0, C1.x, C1.y;
MADR R1, R1, C1.x, C1.y;
ADDR R0, R0, R1;
MULR R0, R0, C2.x;
DP3R R1.x, R0, f[TEX2];
ADDR R2.x, R1.x, R1.x;
MADR R0, R2.x, R0, -f[TEX2];
ADDR R4.x, C1.w, -R1;
POWR R4.x, R4.x, C1.z;
TEX R0, R0, TEX7, LWBE;
TEX R1, R3, TEX1, 2D;
TEX R2, R3, TEX2, 2D;
TEX R3, f[TEX4], TEX5, 2D;
MULR R4.x, R4.x, R3.w;
ADDR R0, R0, R1;
MADR R5, R4.x, -R2, R2;
MADR R0, R4.x, R0, R5;
MULR H0, R0, R3;
MOVH o[COLH], H0; 
END

# Passes = 25 

# Registers = 6 

# Textures = 5 
