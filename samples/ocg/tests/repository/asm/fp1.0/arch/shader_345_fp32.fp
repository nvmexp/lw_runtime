!!FP1.0
DECLARE CCONST0={0.34, 0.45, 0.55, 0.66};
DECLARE CCONST01={0.45, 0.66, 0.55, 0.77};
DECLARE CCONST02={0.32, 0.33, 0.33, 0.55};
DECLARE CCONST03={0.66, 0.55, 0.65, 0.65};
DECLARE CCONST04={0.76, 0.56, 0.57, 0.34};
DECLARE CCONST06={0.34, 0.45, 0.56, 0.67};
TEX R0, f[TEX0], TEX0, 2D;
MADR R2.xyz, f[COL0], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
MADR R3.xyz, R0, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R_SAT R2.w, R2, R3;
DP3R R2.x, f[TEX1], R3;
DP3R R2.y, f[TEX2], R3;
MOVR R0.x, R0.w;
MOVR R0.w, R0.x;
MOVR R0.xyz, f[TEX3];
DP3R R2.z, R0, R3;
DP3R R0.x, R2, R2;
TEX R1, f[TEX4], TEX4, 2D;
MADR R3, R1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R0.y, R2, R3;
MULR R1.xyz, R0.y, {2, 2, 2, 2};
MULR R1.xyz, R2, R1;
MADR R0.xyz, -R3, R0.x, R1;
TEX R1, R0, TEX3, 2D;
MULR R0.xyz, R1, CCONST0;
MULR R2.xyz, R0, R0;
ADDR R3, R2, -R0;
MADR R0.xyz, CCONST01, R3, R0;
DP3R R2.x, R0, CCONST03;
ADDR R1.xyz, R0, -R2.x;
MADR R1.xyz, CCONST02, R1, R2.x;
ADDR R3.w, {1, 1, 1, 1}, -R2;
MULR R1.w, R3, R1;
MULR R1.w, R1, R1;
MULR R1.w, R1, R3.w;
MOVR R0.x, CCONST04.w;
MADR R1.w, R1, CCONST06, R0.x;
MULR H0.xyz, R1, R1.w;
MOVR H0.w, R0.w;
MOVH o[COLH], H0; 
END

# Passes = 24 

# Registers = 4 

# Textures = 5 
