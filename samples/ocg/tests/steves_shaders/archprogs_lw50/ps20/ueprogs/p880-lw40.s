ps_2_0
def c1, 3.000000, 0.500000, 0.000000, 0.000000
dcl t1.rg
dcl t2.rg
dcl t5
dcl_2d s0
dcl_2d s1
mov r0.a, c1.r
mad r0.rg, t2, r0.a, c0
texld r0, r0, s1
texld r1, t1, s0
mad r1.rgb, r0, -c1.g, r1
mul r1.rgb, r1.a, r1
mad_pp r0.rgb, r0, c1.g, r1
mov r0.a, t5.a
mov oC0, r0
