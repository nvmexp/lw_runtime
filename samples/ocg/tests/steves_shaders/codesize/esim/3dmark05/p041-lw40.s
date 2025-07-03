ps_2_0

dcl t0.rg
dcl_2d s0
dcl_2d s1
texld_pp r1, t0, s1
texld_pp r0, t0, s0
mad_sat_pp r1.rgb, r1, c0.r, c0.g
log_pp r1.r, r1.r
log_pp r1.g, r1.g
log_pp r1.b, r1.b
mul_pp r1.rgb, r1, c0.b
exp_pp r1.r, r1.r
exp_pp r1.g, r1.g
exp_pp r1.b, r1.b
mul_pp r1.rgb, r1, c4
dp3_pp r2.r, r0, c1
mad_pp r2.rgb, r2.r, c2, -r0
mad_pp r0.rgb, c2.a, r2, r0
mad_pp r2.rgb, r0, c3, r1
mul_pp r0.rgb, r0, c3
mad_pp r0.rgb, r0, -r1, r2
mad_sat_pp r0.rgb, r0, c5.r, c5.g
log_pp r0.r, r0.r
log_pp r0.g, r0.g
log_pp r0.b, r0.b
mul_pp r0.rgb, r0, c5.b
exp_pp r0.ba, r0.b
exp_pp r0.r, r0.r
exp_pp r0.g, r0.g
mov_pp oC0, r0

