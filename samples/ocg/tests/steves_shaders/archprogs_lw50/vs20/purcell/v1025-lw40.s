vs_2_0
dcl_position0 v0
dcl_texcoord0 v1
add oT1.xy, v1, c[90]
add oT2.xy, v1, -c[90]
add oT3.xy, v1, c[91]
add oT3.zw, v1.xyyx, -c[91].xyyx
add oT4.xy, v1, c[92]
add oT4.zw, v1.xyyx, -c[92].xyyx
add oT5.xy, v1, c[93]
add oT5.zw, v1.xyyx, -c[93].xyyx
add oT6.xy, v1, c[94]
add oT6.zw, v1.xyyx, -c[94].xyyx
add oT7.xy, v1, c[95]
add oT7.zw, v1.xyyx, -c[95].xyyx
mov oPos.xyz, v0
mov oPos.w, c[0].y
mov oT0.xy, v1
