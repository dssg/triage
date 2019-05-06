## -*- docker-image-name: "triage-v4.0.0" -*-

FROM python:3.7-alpine

LABEL triage.version="v4.0.0" \
      triage.from="cli" \
      creator="Center for Data Science and Public Policy (DSaPP)" \
      maintainer="Adolfo De Unánue <adolfo@uchicago.edu>"

# the lapack package is only in the community repository
RUN echo "http://dl-4.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories

RUN apk --update add --no-cache \
        build-base \
        lapack-dev \
        gcc \
        git \
        postgresql-dev \
        libpng-dev freetype-dev \
        libpq


RUN apk add --no-cache --virtual .build-deps \
        lapack-dev \
        gfortran \
        musl-dev \
        g++

RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir triage

WORKDIR triage

# removing dependencies
RUN apk del .build-deps

ENTRYPOINT [ "triage" ]
