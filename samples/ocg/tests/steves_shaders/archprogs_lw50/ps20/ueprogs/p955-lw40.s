ps_2_0
def c2, 2.000000, -1.000000, -0.500000, 0.500000
dcl t1.rg
dcl t5
dcl t7.rgb
dcl_2d s0
dcl_2d s1
texld r1, t1, s0
add r0, r1.a, c2.b
mad r2.rgb, c2.r, r1, c2.g
texkill r0
texld r0, t1, s1
nrm r1.rgb, r2
mul r1.rgb, r1, c0.r
nrm r2.rgb, t7
dp3 r1.r, r1, r2
mad r0.a, r1.r, c2.a, c2.a
mul r0.a, r0.a, r0.a
mul r0.rgb, r0, r0.a
mul r0.rgb, r0, c1
mov r0.a, t5.a
mov oC0, r0
