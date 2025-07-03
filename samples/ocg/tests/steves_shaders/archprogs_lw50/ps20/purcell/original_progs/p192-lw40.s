ps_2_0
def c2, 0.500000, 0.500000, 0.250000, 0.000000
def c3, 6.283190, -0.000000, 0.000025, -3.141590
def c4, -0.001389, -0.500000, 1.000000, 0.041667
def c5, 0.000000, 1.200000, 0.400000, 20.000000
def c6, 2.000000, -1.000000, 0.000000, 0.000000
dcl t0.rgb
dcl t1.rg
dcl t3
dcl t4.rgb
dcl_volume s0
dcl_2d s1
mad r7.rgb, t0, c2.r, c2.r
texld r2, r7, s0
texld r9, t1, s1
mad r9.a, c6.r, r2.r, c6.g
mad r9.a, t0.r, c0.a, r9.a
mad r9.a, r9.a, c2.g, c2.b
frc r9.a, r9.a
mad r9.a, r9.a, c3.r, c3.a
mul r9.a, r9.a, r9.a
mad r6.a, r9.a, c3.g, c3.b
mad r8.a, r9.a, r6.a, c4.r
mad r10.a, r9.a, r8.a, c4.a
mad r0.a, r9.a, r10.a, c4.g
mad r9.a, r9.a, r0.a, c4.b
abs r9.a, r9.a
log r9.a, r9.a
mul r9.a, r9.a, c2.r
exp r9.a, r9.a
mov r2.rgb, -c1
add r4.rgb, r2, c0
mad r6.rgb, r4, r9.a, c1
mul r1.rgb, r9, r6
mul r8.rgb, r1, c5.g
dp4 r8.a, t3, t3
rsq r8.a, r8.a
mul r3.rgb, r8.a, t3
dp3 r8.a, t4, r3
cmp r8.a, r8.a, r8.a, c5.r
log r8.a, r8.a
mul r8.a, r8.a, c5.a
exp r8.a, r8.a
mul r8.a, r8.a, c5.b
add r8.a, r8.a, c4.b
mul r10.rgb, r8, r8.a
mov r10.a, c4.b
mov oC0, r10
