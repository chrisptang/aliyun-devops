FROM xianmu-registry-registry.cn-hangzhou.cr.aliyuncs.com/base/pyodps-pandas:3.8-slim as custom-python-base

ADD requirements.txt ./requirements.txt

# RUN apt-get update && apt-get install -y gcc && apt-get clean

RUN pip install --index-url=https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host=mirrors.aliyun.com --default-timeout=30 -r requirements.txt

WORKDIR /app

# Stage 2: Final deployment image
FROM custom-python-base

ADD run_command.sh ./run_command.sh
RUN chmod +x ./run_command.sh

ENV HOURS_TO_RUN "8"
ENV MINUTES_INTERVAL "240"
ENV MINUTES_INTERVAL_SMALL "15"
ENV MYSQL_HOST "host.docker.internal"
ENV MYSQL_PORT "3306"
ENV MYSQL_USER_NAME "root"
ENV MYSQL_PASSWORD "peng_mbp13"
ENV PYTHON_RUN_INTERVAL "90"
ENV ROUNDS_TO_RUN "2"
ENV INIT_LAST_7DAYS_DATA "true"
ENV RUN_CONTINUOUSLY "true"

ENV TZ Asia/Shanghai

ADD ./阿里云SLS日志分析-前端日志到MySQL.py ./notebook.py

ENTRYPOINT ["python", "./notebook.py"]