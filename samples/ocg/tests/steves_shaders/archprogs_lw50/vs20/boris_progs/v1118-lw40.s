vs_2_0
def c[1], 0.816497, -0.408248, 0.707107, -0.707107
def c[3], 0.577350, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v4
mov r0.w, c[0].y
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp4 oPos.x, r0, c[8]
dp3 r3.x, v1, c[42]
dp3 r2.x, v4, c[42]
dp3 r3.y, v1, c[43]
dp3 r3.z, v1, c[44]
dp3 r2.y, v4, c[43]
dp3 r2.z, v4, c[44]
dp4 oPos.y, r0, c[9]
mul r1.xyz, r3.zxyw, r2.yzxw
dp4 oPos.w, r0, c[11]
mad r1.xyz, r3.yzxw, r2.zxyw, -r1
dp4 r8.y, r0, c[10]
mul r1.xyz, r1, v4.w
add oT3.xyz, -r0, c[2]
dp3 r4.y, c[1].xyyw, r1
dp3 r4.x, c[1].xyyw, r2
dp3 r4.z, c[1].xyyw, r3
slt r0.xyz, r4, c[0].x
mul r5.xyz, r4, r4
mova a0.xyw, r0.xyz
mul r0.xy, r2.yzzw, c[1].zwzw
mul r4.xyz, r5.y, c[a0.y+23]
add r0.x, r0.y, r0.x
mul r7.xy, r3.yzzw, c[1].zwzw
mul r6.xy, r1.yzzw, c[1].zwzw
add r0.z, r7.y, r7.x
add r0.y, r6.y, r6.x
mad r6.xyz, r5.x, c[a0.x+21], r4
slt r4.xyz, r0, c[0].x
mul r0.xyz, r0, r0
mova a0.xyz, r4
mad oFog, -r8.y, c[16].w, c[16].x
mul r4.xyz, r0.y, c[a0.y+23]
mov oPos.z, r8.y
mad r4.xyz, r0.x, c[a0.x+21], r4
mad oD0.xyz, r5.z, c[a0.w+25], r6
mad oD1.xyz, r0.z, c[a0.z+25], r4
dp3 r0.x, c[3].x, r2
mov oT4.x, r2.x
mov oT5.x, r2.y
mov oT6.x, r2.z
dp3 r0.z, c[3].x, r3
dp3 r0.y, c[3].x, r1
mov oT4.z, r3.x
slt r2.xyz, r0, c[0].x
mul r0.xyz, r0, r0
mova a0.xyz, r2
mov oT5.z, r3.y
mul r2.xyz, r0.y, c[a0.y+23]
mov oT6.z, r3.z
mad r2.xyz, r0.x, c[a0.x+21], r2
mad oT7.xyz, r0.z, c[a0.z+25], r2
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
mov oT4.y, r1.x
mov oT5.y, r1.y
mov oT6.y, r1.z
mov oD0.w, c[0].x
mov oD1.w, c[0].x
