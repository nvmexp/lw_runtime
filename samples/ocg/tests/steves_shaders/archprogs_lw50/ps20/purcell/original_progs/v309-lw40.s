vs_1_1
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
m4x4 oPos, v0, c[0]
mov r0, v1
add r0, r0, -v0
add r1, c[4], -v0
mul r2, r0.yzxw, r1.zxyw
mad r2, -r1.yzxw, r0.zxyw, r2
mul r3, r2.yzxw, r0.zxyw
mad r3, -r0.yzxw, r2.zxyw, r3
dp3 r3.w, r3, r3
rsq r3.w, r3.w
mul r3.xyz, r3, r3.w
dp3 r0.w, r1, r1
rsq r0.w, r0.w
mul r1.xyz, r1, r0.w
dp3 r1.x, r3, r1
mul oT0.xy, r1.xw, v2.xy
mov oT0.zw, v0.w
