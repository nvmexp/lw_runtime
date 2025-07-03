ps_2_0
def c4, 0.333333, 0.000000, 0.000000, 0.000000
dcl t3.rgb
dcl t6.rgb
dcl_lwbe s1
dp3 r0.r, t6, t6
rcp r0.a, r0.r
mov r0.rgb, t6
dp3 r1.r, r0, t3
mul r0.a, r0.a, r1.r
add r0.a, r0.a, r0.a
mad r0.rgb, r0.a, r0, -t3
texld_pp r0, r0, s1
mul_pp r1.rgb, r0, c0
mad_pp r0.rgb, r1, r1, -r1
mad_pp r1.rgb, c2, r0, r1
dp3_pp r2.r, r1, c4.r
lrp_pp r0.rgb, c3, r1, r2.r
mov_pp r0.a, c1.a
mov_pp oC0, r0
