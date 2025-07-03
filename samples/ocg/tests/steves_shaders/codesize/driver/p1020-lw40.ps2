ps_2_0
def c2, -0.500000, 0.500000, 0.000000, 0.000000
def c3, 0.350000, 0.500000, 0.300000, 0.000000
def c4, 0.000000, 0.000000, 1.000000, 0.000000
dcl t1.rg
dcl t5
dcl t7.rgb
dcl_2d s0
texld r1, t1, s0
add r0, r1.a, c2.r
nrm r3.rgb, t7
mov r1.a, c0.r
mul r2.rgb, r1.a, c4
dp3 r2.r, r2, r3
mul r1.rgb, r1, c3
mad r1.a, r2.r, c2.g, c2.g
texkill r0
mul r0.a, r1.a, r1.a
mul r0.rgb, r1, r0.a
mul r0.rgb, r0, c1
mov r0.a, t5.a
mov oC0, r0
