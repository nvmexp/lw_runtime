vs_2_0
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v3
dcl_binormal0 v4
dp4 r0.x, v0, c[4]
dp4 r0.y, v0, c[5]
dp4 r2.y, v0, c[7]
add r1.xy, r0, r2.y
mul oT2.xy, r1, c[0].w
mov r0.z, -r0.y
mov oPos.xy, r0
add r0.xy, r0.xzzw, r2.y
mul oT5.xy, r0, c[0].w
dp3 r0.x, v3, c[8]
mul oT3.x, r0.x, c[90].x
mul oT6.x, r0.x, c[93].x
dp3 r0.x, v4, c[8]
mul oT3.y, r0.x, c[90].x
mul oT6.y, r0.x, c[93].x
dp3 r0.x, v3, c[9]
mul oT4.x, r0.x, -c[90].x
mul oT7.x, r0.x, -c[93].x
dp3 r0.x, v4, c[9]
mul oT4.y, r0.x, -c[90].x
mul oT7.y, r0.x, -c[93].x
dp4 r0.y, v0, c[6]
mad oFog, -r0.y, c[16].w, c[16].x
mov oPos.z, r0.y
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
add r0.xyz, -r0, c[2]
dp3 oT1.x, r0, v3
dp3 oT1.y, r0, v4
dp3 oT1.z, r0, v1
dp4 oT0.x, v2, c[91]
dp4 oT0.y, v2, c[92]
mov oPos.w, r2.y
mov oT2.z, r2.y
mov oT5.z, r2.y
