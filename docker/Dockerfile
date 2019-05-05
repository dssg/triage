## -*- docker-image-name: "triage-v4.0.0" -*-

FROM python:3.6.2-alpine

LABEL triage.version="v4.0.0" \
      triage.from="cli" \
      creator="Center for Data Science and Public Policy (DSaPP)" \
      maintainer="Adolfo De Unánue <adolfo@uchicago.edu>"

RUN apk update \
        && apk add \
        build-base \
        git \
        postgresql-dev \
        libpq


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir triage

WORKDIR triage

ENTRYPOINT [ "triage" ]
