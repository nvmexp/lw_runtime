vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_color1 v2
dcl_texcoord0 v3
dcl_tangent0 v5
dp3 r2.x, v1, c[42]
dp3 r2.y, v1, c[43]
dp3 r2.z, v1, c[44]
dp3 r0.x, v5, c[42]
dp3 r0.y, v5, c[43]
dp3 r0.z, v5, c[44]
mul r1.xyz, r2.zxyw, r0.yzxw
mad r1.xyz, r2.yzxw, r0.zxyw, -r1
mov oT6.xyz, r2
mov oT4.xyz, r0
mov r0.w, c[0].y
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
mul oT5.xyz, r1, v5.w
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r1.y, r0, c[10]
add oT3.xyz, -r0, c[2]
mad oFog, -r1.y, c[16].w, c[16].x
mul r0.xyz, v2, c[1].y
log r0.x, r0.x
log r0.y, r0.y
log r0.z, r0.z
rcp r0.w, c[1].x
mov oPos.z, r1.y
mul r0.xyz, r0, r0.w
exp oD0.x, r0.x
exp oD0.y, r0.y
exp oD0.z, r0.z
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
