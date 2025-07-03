ps_3_0

def c8, 0.00000000, 1.00000000, 2.00000000, -1.00000000 ; 0x000000 0x3f800000 0x40000000 0xbf800000
def c9, 16.00000000, 0.00000000, 0.00000000, 0.00000000 ; 0x41800000 0x000000 0x000000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord4_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_2d s5
cmp r0.w, -v2.z, c8.x, c8.y
if_ne r0.w, -r0.w
cmp r0.w, -v3.w, c8.x, c8.y
if_ne r0.w, -r0.w
dp3 r0.x, v2, c0
dp3 r0.y, v2, c1
dp3 r0.z, v2, c2
dp3 r0.w, r0, r0
add_sat_pp r2.w, -r0.w, c8.y
rcp r0.z, v3.w
mov r0.w, c8.y
mul r0.xyz, r0.z, v3
texldl_pp r1, r0, s1
mov r0, c3
mul r0, r0, c4.x
mul r0, r2.w, r0
texldp r2, v3, s0
mul r0, r0, r2
mul r4, r1.x, r0
mov_pp r0, r4
else
mov r1, c8.x
mov_pp r0, r1
mov_pp r4, r1.x
endif
dp3_pp r1.w, v1, v1
rsq_pp r1.w, r1.w
nrm_pp r6.xyz, v2
mad_pp r1.xyz, v1, r1.w, r6
nrm_pp r2.xyz, r1
texld r1, v0.xyx, s4
mad_pp r5.xyz, c8.z, r1.wyzw, c8.w
dp3_pp r2.x, r5, r2
texld_pp r3, v0.xyx, s3
mul_pp r2.w, r3.w, c7.x
mov r2.yz, c8.x
texldl_pp r1, r2.xwzy, s5
mul_pp r1, r4, r1.x
mul_pp r2.xyz, r3, c6
mul_pp r1, r2, r1
dp3_pp r2.z, r5, r6
mul_sat_pp r2.w, r2.z, c9.x
mov_sat_pp r3.w, r2.z
mul_pp r2, r1, r2.w
mul_pp r0, r0, r3.w
texld_pp r1, v0.xyx, s2
mul_pp r1, r1, c5
mad oC0, r0, r1, r2
else
mov oC0, c8.x
endif
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex ffdd -texrange 7ff -TexShadowMap 2
