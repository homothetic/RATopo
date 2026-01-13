#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

CONF=$1
GPUS=$2
PORT=${PORT:-28510}

WORK_DIR=work_dirs/${CONF}
CONFIG=projects/configs/${CONF}.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic --autoscale-lr ${@:3} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
