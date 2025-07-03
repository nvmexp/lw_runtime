vs_1_1
dcl_position0 v0
dcl_texcoord0 v7
dcl_texcoord1 v8
dp4 r0.x, v0, c[4]
dp4 r0.y, v0, c[5]
dp4 r0.z, v0, c[6]
dp4 r0.w, v0, c[7]
mov oPos, r0
dp4 r1.z, v0, c[44]
add r2.xy, c[2].wz, -r1.z
max r2.x, r2.x, c[0].x
rcp r2.z, r2.y
mul r2.w, r2.x, r2.z
mul r2.w, r2.w, r0.z
mad oFog, -r2.w, c[16].w, c[16].y
dp4 oT0.x, v7, c[90]
dp4 oT0.y, v7, c[91]
dp4 oT1.x, v7, c[92]
dp4 oT1.y, v7, c[93]
mov oT2, v8
