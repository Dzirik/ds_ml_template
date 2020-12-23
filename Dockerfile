FROM python:3.8.3

ENV ENV_DASH_DEBUG_MODE True
ENV ENV_DASH_DOCKER True

RUN mkdir /dash
COPY requirements.txt /dash/

WORKDIR /dash
RUN set -ex && \
    pip install -r requirements.txt

EXPOSE 8050

COPY ./ ./

CMD ["python", "index.py"]