vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v3
dcl_binormal0 v4
dp4 r0.x, v0, c[4]
dp4 r0.y, v0, c[5]
dp4 r1.w, v0, c[7]
add r1.xy, r0, r1.w
mov r0.z, -r0.y
mul oT2.xy, r1, c[0].w
add r0.zw, r0.xyzx, r1.w
mov oPos.xy, r0
mul oT2.zw, r0, c[0].w
dp4 r1.y, v0, c[6]
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
mad oFog, -r1.y, c[16].w, c[16].x
add r0.xyz, -r0, c[2]
mov oPos.z, r1.y
dp3 oT1.x, r0, v3
dp3 oT1.y, r0, v4
dp3 oT1.z, r0, v1
dp4 oT0.x, v2, c[91]
dp4 oT0.y, v2, c[92]
mov oPos.w, r1.w
mov oT3.x, r1.w
