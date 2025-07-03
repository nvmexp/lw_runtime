vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_color1 v2
dcl_texcoord0 v3
dcl_tangent0 v5
dp3 r0.x, v5, c[42]
dp3 r0.y, v5, c[43]
dp3 r0.z, v5, c[44]
dp3 r2.x, v1, c[42]
dp3 r2.y, v1, c[43]
dp3 r2.z, v1, c[44]
mul r1.xyz, r0.yzxw, r2.zxyw
mad r1.xyz, r2.yzxw, r0.zxyw, -r1
mov oT4.xyz, r0
mov r0.w, c[0].y
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
mul oT5.xyz, r1, v5.w
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 r1.y, r0, c[10]
dp4 oPos.w, r0, c[11]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
add r1.xyz, -r0, c[29]
add oT3.xyz, -r0, c[2]
dp3 r3.z, r1, r1
rsq r0.w, r3.z
add r0.xyz, -r0, c[34]
dst r3.xy, r3.z, r0.w
mul r1.xyz, r1, r0.w
mov r6.z, c[0].y
mad r0.w, r3.z, -c[31].w, r6.z
dp3 r3.x, c[31], r3
max r1.w, r0.w, c[0].x
rcp r0.w, r3.x
min r1.w, r1.w, c[0].y
mul r1.w, r0.w, r1.w
mul r3.xyz, v2, c[1].y
log r3.x, r3.x
log r3.y, r3.y
log r3.z, r3.z
slt r4.xyz, r2, c[0].x
mova a0.xyz, r4
mul r4.xyz, r2, r2
rcp r0.w, c[1].x
mul r5.xyz, r4.y, c[a0.y+23]
mul r3.xyz, r3, r0.w
mad r5.xyz, r4.x, c[a0.x+21], r5
exp r3.x, r3.x
exp r3.y, r3.y
exp r3.z, r3.z
mad r4.xyz, r4.z, c[a0.z+25], r5
add r4.xyz, r3, r4
dp3 r1.x, r2, r1
max r0.w, r1.x, c[0].x
dp3 r3.z, r0, r0
mul r1.xyz, r0.w, c[27]
rsq r0.w, r3.z
mad r1.xyz, r1, r1.w, r4
mul r0.xyz, r0, r0.w
dst r3.xy, r3.z, r0.w
dp3 r3.x, c[36], r3
mad r1.w, r3.z, -c[36].w, r6.z
rcp r0.w, r3.x
max r1.w, r1.w, c[0].x
min r1.w, r1.w, c[0].y
dp3 r0.x, r2, r0
mul r0.w, r0.w, r1.w
max r1.w, r0.x, c[0].x
mov oT6.xyz, r2
mul r0.xyz, r1.w, c[32]
mad oD0.xyz, r0, r0.w, r1
mul r0.xy, v3, c[90]
add r0.w, r0.y, r0.x
mul r0.xy, v3, c[91]
mov oT0.x, r0.w
add r0.z, r0.y, r0.x
mov oT0.y, r0.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r0.z
mov oT2.y, r0.z
mov oD0.w, c[0].x
mov oD1, c[0].x
mov oT7.xyz, c[0].x
