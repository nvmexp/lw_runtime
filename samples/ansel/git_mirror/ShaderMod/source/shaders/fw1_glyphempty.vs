struct GSIn {
	float3 PositionIndex : POSITIONINDEX;
	float4 GlyphColor : GLYPHCOLOR;
};

GSIn main(GSIn Input) {
	return Input;
}
