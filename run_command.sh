#!/bin/bash

if [[ "${INIT_LAST_7DAYS_DATA:-false}" == "true" ]]; then
    # 初始化最近七天的数据
    # 先保存系统设置；
    export RUN_CONTINUOUSLY="false"
    rounds="${ROUNDS_TO_RUN:-2}"
    interval="${HOURS_INTERVAL_SMALL:-1}"

    export ROUNDS_TO_RUN="43" # 4小时一个间隔、跑7天的数据，所以是 42个周期
    export HOURS_INTERVAL_SMALL="4"

    echo "即将初始化数据, ROUNDS_TO_RUN:${ROUNDS_TO_RUN}"
    python3 notebook.py

    export ROUNDS_TO_RUN="${rounds}" # 还原回去
    export HOURS_INTERVAL_SMALL="${interval}"
    export RUN_CONTINUOUSLY="true"

fi