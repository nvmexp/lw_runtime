vs_2_0
dcl_position0 v0
dcl_texcoord0 v1
mov r0.w, c[0].y
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r1.y, r0, c[10]
add oT3.xyz, -r0, c[2]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
mul r1.xy, v1, c[90]
mul r0.xy, v1, c[91]
add r0.w, r1.y, r1.x
add r0.z, r0.y, r0.x
mov oT0.x, r0.w
mov oT0.y, r0.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r0.z
mov oT2.y, r0.z
mov oT4.xyz, c[0].x
mov oT5.xyz, c[0].x
mov oT6.xyz, c[0].x
mov oD0, c[0].x
mov oD1, c[0].x
mov oT7.xyz, c[0].x
