ps_2_0

dcl t0.rg
dcl_2d s0
texld_pp r0, t0, s0
mad_sat_pp r0.rgb, r0, c0.r, c0.g
log_pp r0.r, r0.r
log_pp r0.g, r0.g
log_pp r0.b, r0.b
mul_pp r0.rgb, r0, c0.b
exp_pp r1.r, r0.r
exp_pp r1.g, r0.g
exp_pp r1.b, r0.b
dp3_pp r0.r, r1, c1
mad_pp r0.rgb, r0.r, c2, -r1
mad_pp r1.rgb, c2.a, r0, r1
mov_pp r0.rgb, r1
mov_pp r0.a, r1.b
mov_pp oC0, r0

