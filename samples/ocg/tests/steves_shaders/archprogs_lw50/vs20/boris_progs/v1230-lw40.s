vs_2_0
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord1 v2
dcl_color0 v3
mov r0.xyz, v0
mov r0.w, c[0].y
dp4 oPos.x, r0, c[4]
dp4 oPos.y, r0, c[5]
dp4 oPos.w, r0, c[7]
dp4 r1.x, r0, c[42]
dp4 r1.y, r0, c[43]
dp4 r1.z, r0, c[44]
dp4 r3.y, r0, c[6]
add oT4.xyz, -r1, c[2]
mul r1.xy, v1, c[90]
add r0.xy, -r1.z, c[2].wzzw
add oT0.x, r1.y, r1.x
mul r2.xy, v1, c[91]
mul r1.xy, v1, c[96]
add oT0.y, r2.y, r2.x
add oT0.z, r1.y, r1.x
mul r2.xy, v1, c[97]
mul r1.xy, v1, c[92]
add oT0.w, r2.y, r2.x
add oT1.x, r1.y, r1.x
max r0.w, r0.x, c[0].x
rcp r1.w, r0.y
mul r0.xy, v1, c[93]
mul r0.w, r0.w, r1.w
add oT1.y, r0.y, r0.x
mul r0.w, r3.y, r0.w
mov oPos.z, r3.y
mad oFog, -r0.w, c[16].w, c[16].y
mul oD0, v3, c[38]
mov oT1.zw, c[0].x
mov oT2, c[0].x
mov oT3.xy, v2
mov oT3.zw, c[0].x
mov oT5.xyz, c[0].x
mov oT6.xyz, c[0].x
mov oT7.xyz, c[0].x
