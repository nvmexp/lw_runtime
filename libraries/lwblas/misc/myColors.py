intel = "#127cc1"

qualitative1 = [
"#66c2a5",
"#fc8d62",
"#8da0cb",
"#e78ac3",
"#a6d854",
"#e5c494",
"#b3b3b3",
"#ffd92f"
]

types = {
        "fp32"      :qualitative1[0],
        "fp16"      :qualitative1[1],
        "bf16"      :qualitative1[1],
        "fp64"      :qualitative1[2],
        "fp64+fp32" :qualitative1[3],
        "fp32+fp16" :qualitative1[4],
        "fp32+tf32" :qualitative1[5],
        }

versions = {
    "lwTENSOR 1.2.1": qualitative1[4],
    "lwTENSOR 1.2.2": qualitative1[0]
}
