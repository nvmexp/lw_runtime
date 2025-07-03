ps_2_0
def c1, 4.000000, 2.000000, -1.000000, 0.500000
dcl t1.rg
dcl t5
dcl t7.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s2
mul r0.rg, t1, c1.r
texld r2, r0, s1
texld r1, t1, s0
texld r0, t1, s2
mad r2.rgb, c1.g, r2, c1.b
mad r1.rgb, c1.g, r1, r2
add r2.rgb, r1, c1.b
nrm r1.rgb, r2
nrm r2.rgb, t7
dp3 r1.r, r1, r2
mad r0.a, r1.r, c1.a, c1.a
mul r0.a, r0.a, r0.a
mul r0.rgb, r0, r0.a
mul r0.rgb, r0, c0
mov r0.a, t5.a
mov oC0, r0
