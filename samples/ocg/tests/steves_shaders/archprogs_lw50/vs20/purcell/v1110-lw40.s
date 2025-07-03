vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v4
dp3 r0.x, v4, c[42]
dp3 r0.y, v4, c[43]
dp3 r0.z, v4, c[44]
dp3 r3.x, v1, c[42]
dp3 r3.y, v1, c[43]
dp3 r3.z, v1, c[44]
mul r1.xyz, r0.yzxw, r3.zxyw
mad r1.xyz, r3.yzxw, r0.zxyw, -r1
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
add r1.xyz, -r0, c[29]
dp3 r4.x, r1, r1
mov r4.y, c[0].y
mov r2.xz, c[0].y
rsq r2.y, r4.x
add oT3.xyz, -r0, c[2]
mul r0.xyz, r4.yxxw, r2
mov r2.z, c[0].y
mad r0.w, r4.x, -c[31].w, r2.z
dp3 r0.x, r0, c[31]
mul r1.xyz, r1, r2.y
rcp r1.w, r0.x
max r0.w, r0.w, c[0].x
dp3 r0.x, c[28], -r1
min r2.w, r0.w, c[0].y
add r0.w, r0.x, -c[30].z
mul r1.w, r1.w, r2.w
mul r0.w, r0.w, c[30].w
mul r0.xyz, r1.w, c[27]
max r1.w, r0.w, c[0].x
dp3 r4.x, r3, r1
pow r0.w, r1.w, c[30].x
min r0.w, r0.w, c[0].y
slt r1.xyz, r3, c[0].x
mova a0.xyz, r1
mul r1.xyz, r3, r3
mul r0.xyz, r0, r0.w
mul r2.xyz, r1.y, c[a0.y+23]
mov oT6.xyz, r3
mad r2.xyz, r1.x, c[a0.x+21], r2
mad r1.xyz, r1.z, c[a0.z+25], r2
max r0.w, r4.x, c[0].x
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
