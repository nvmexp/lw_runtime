import argparse
import subprocess
import os
import sys

import generateContractionLwtlass

from collections import OrderedDict


KERNEL_TEMP_FILE="misc/kernels/temp.kernels"

def handleArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", type=str, nargs="*")
    return parser.parse_args()


def filterFeature(featureList, featureName):
    return list(filter(lambda feature: not feature.startswith(
        featureName + ":"), featureList))


def generate_sass_features(kernels_file):
    print(f"Generating sass features for {kernels_file}")

    data_type = kernels_file.split(".")[-3]
    arch = kernels_file.split(".")[-2][2:]

    print(f"data_type={data_type}, arch={arch}")

    new_kernels = []

    with open(kernels_file, "r") as f:
        kernels = f.readlines()

    for kernel in kernels:
        if kernel.startswith("#"):
            new_kernels.append(kernel.rstrip())
            continue
        kernel = kernel.rstrip()
        print(f"Parsing kernel: {kernel}")
        os.system(f"echo \"{kernel}\" > {KERNEL_TEMP_FILE}")
        generateContractionLwtlass.main(KERNEL_TEMP_FILE, False, True, None)
        result = subprocess.run(
            ["./misc/compileContraction.sh", data_type, arch], stdout=subprocess.PIPE)
        output = result.stdout.decode("utf-8").split("\n")

        wait_schedule = None
        avg_lds = None
        avg_ldg = None
        avg_anti = None

        for line in output:
            #if "WaitSchedule" in line:
            #    wait_schedule = int(float(line.split(" ")[-1].rstrip()))
            if "LDS" in line:
                avg_lds = int(float(line.split(" ")[-1].rstrip()))
            if "LDG" in line:
                avg_ldg = int(float(line.split(" ")[-1].rstrip()))
            if "Antidep" in line:
                avg_anti = int(float(line.split(" ")[-1].rstrip()))

        #assert(wait_schedule is not None)
        assert(avg_lds is not None)
        #assert(avg_ldg is not None)
        assert(avg_anti is not None)

        result = subprocess.run(["python3",
                                 "misc/detectLocalMemoryUsage.py",
                                 f"build/keep/lwtensor_base_objs_sm{arch}"],
                                stdout=subprocess.PIPE)
        output = result.stdout.decode("utf-8").split("\n")
        count = 0
        while count < len(output) and f"sm{arch}_{data_type}.lwbin" not in output[count]:
            count += 1
        lmem = int(output[count + 1].split(" ")[-1])

        features = OrderedDict(map(lambda s : s.split(":", 2), kernel.split(";")))

        features["lmem"] = lmem
        #features["wa"] = wait_schedule
        features["ls"] = avg_lds
        #features["lg"] = avg_ldg
        features["la"] = avg_anti

        new_kernels.append(";".join([f"{key}:{values}" for key, values in features.items()]))

    with open(kernels_file, "w") as f:
        for kernel in new_kernels:
            f.write(kernel + "\n")

    generateContractionLwtlass.main(kernels_file, True, False, None)


def main():
    args = handleArguments()

    for kernels_path in args.kernels:
        generate_sass_features(kernels_path)

    subprocess.run(["rm", KERNEL_TEMP_FILE])


if __name__ == "__main__":
    main()
