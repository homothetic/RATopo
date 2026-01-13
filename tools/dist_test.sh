#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

CONF=$1
GPUS=$2
PORT=${PORT:-28510}

WORK_DIR=work_dirs/${CONF}
CONFIG=projects/configs/${CONF}.py
CHECKPOINT=${WORK_DIR}/latest.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 --out ${@:3} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log

# --show --input