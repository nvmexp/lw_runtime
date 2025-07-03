vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v4
dp3 r0.x, v4, c[42]
dp3 r0.y, v4, c[43]
dp3 r0.z, v4, c[44]
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
mul oT5.xyz, r1, v4.w
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 r1.y, r0, c[10]
dp4 oPos.w, r0, c[11]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
add oT3.xyz, -r0, c[2]
mov r5.y, c[0].y
add r3.xyz, -r0, c[29]
mov r4.xz, c[0].y
dp3 r5.x, r3, r3
add r0.xyz, -r0, c[34]
rsq r4.y, r5.x
mul r1.xyz, r5.yxxw, r4
mov r5.z, c[0].y
mad r1.w, r5.x, -c[31].w, r5.z
mul r3.xyz, r3, r4.y
dp3 r1.x, r1, c[31]
rcp r0.w, r1.x
max r1.w, r1.w, c[0].x
min r1.w, r1.w, c[0].y
dp3 r1.x, c[28], -r3
mul r1.w, r0.w, r1.w
add r0.w, r1.x, -c[30].z
mul r1.xyz, r1.w, c[27]
mul r0.w, r0.w, c[30].w
dp3 r5.x, r2, r3
max r1.w, r0.w, c[0].x
pow r0.w, r1.w, c[30].x
slt r3.xyz, r2, c[0].x
mova a0.xyz, r3
mul r3.xyz, r2, r2
min r0.w, r0.w, c[0].y
mul r4.xyz, r3.y, c[a0.y+23]
mul r1.xyz, r1, r0.w
mad r4.xyz, r3.x, c[a0.x+21], r4
mad r4.xyz, r3.z, c[a0.z+25], r4
dp3 r3.z, r0, r0
max r1.w, r5.x, c[0].x
rsq r0.w, r3.z
mad r1.xyz, r1, r1.w, r4
mul r0.xyz, r0, r0.w
dst r3.xy, r3.z, r0.w
dp3 r3.x, c[36], r3
mad r1.w, r3.z, -c[36].w, r5.z
rcp r0.w, r3.x
max r1.w, r1.w, c[0].x
min r1.w, r1.w, c[0].y
dp3 r0.x, r2, r0
mul r0.w, r0.w, r1.w
max r1.w, r0.x, c[0].x
mov oT6.xyz, r2
mul r0.xyz, r1.w, c[32]
mad oD0.xyz, r0, r0.w, r1
mul r0.xy, v2, c[90]
add r0.w, r0.y, r0.x
mul r0.xy, v2, c[91]
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
