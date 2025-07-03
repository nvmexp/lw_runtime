if [ "$#" -eq "1" ]; then
    exelwtable=$1
else
    echo "$0: Incorrect number of arguments!"
    exit 1
fi
TARGET=10.0.0.21
HOST=10.0.0.1
TEST=$(basename $exelwtable)
DIR=$(dirname $exelwtable)
echo "Mounting on target and running the test..."
sshpass -p root ssh -tt root@${TARGET} "fs-nfs3 "${HOST}:${DIR}"       "${DIR}"; cd "${DIR}"; ./"${TEST}
echo "Exelwtion complete"
