ps_2_0
def c2, 2.000000, -1.000000, 12.000000, 0.000000
def c3, 15.000000, 0.000000, 0.000000, 0.000000
dcl t0.rg
dcl t1.rg
dcl t5.rgb
dcl t6.rgb
dcl t7
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_2d s5
texld r0, t1, s1
mad r5.rgb, c2.r, r0, c2.g
mul r1.rg, t1, c2.b
rcp r0.a, t7.a
mul r0.rg, r0.a, t7
mad r0.rg, r0, c0, c0.abgr
texld r4, r1, s0
texld r3, t1, s3
texld r2, t1, s2
texld r0, r0, s4
texld r1, t0, s5
mad r4.rgb, c2.r, r4, r5
add r5.rgb, r4, c2.g
dp3 r6.r, t6, t6
nrm r4.rgb, r5
rsq r0.a, r6.r
mul r5.rgb, r0.a, t6
dp3 r5.r, r4, r5
mul r5.rgb, r4, r5.r
add r5.rgb, r5, r5
mad r6.rgb, t6, -r0.a, r5
nrm r5.rgb, t5
dp3 r6.r, r6, r5
max r1.a, r6.r, c2.a
dp3 r4.r, r4, r5
pow r0.a, r1.a, c3.r
mul r3.rgb, r3, r0.a
max r0.a, r4.r, c2.a
mad r2.rgb, r2, r0.a, r3
mul r0.rgb, r0, r1.r
mul r0.rgb, r2, r0
mul r0.rgb, r0, c1
mov r0.a, c2.a
mov oC0, r0
