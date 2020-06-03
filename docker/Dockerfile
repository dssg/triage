## -*- docker-image-name: "triage" -*-

FROM python:3.7-alpine AS development

LABEL creator="Center for Data Science and Public Policy (DSaPP)" \
      maintainer="Adolfo De Unánue <adolfo@cmu.edu>"

ENV PACKAGES="\
    dumb-init \
    musl \
    libc6-compat \
    linux-headers \
    build-base \
    bash \
    git \
    ca-certificates \
    freetype \
    freetype-dev \
    libgfortran \
    libgcc \
    libstdc++ \
    lapack-dev \
    postgresql-dev \
    libpng-dev \
    openblas \
    tcl \
    tk \
    libpq \
    gcc \
    g++ \
    git \
    curl \
    libffi-dev \
    jpeg-dev \ 
    postgresql \
    "


RUN apk add --virtual build-runtime \
     build-base python-dev openblas-dev lapack-dev freetype-dev pkgconfig gfortran g++ gcc git curl  \
     && ln -s /usr/include/locale.h /usr/include/xlocale.h \
     && pip install --upgrade pip setuptools \
     && rm -r /root/.cache \
        && apk add --no-cache --virtual build-dependencies $PACKAGES

#RUN pip install --no-cache-dir -r requirements.txt \
#    && apk del build-runtime \
#    && rm -rf /var/cache/apk/*

RUN mkdir triage

WORKDIR triage

ENV SHELL=/bin/bash
# Defaults to be overruled by launch script (docker run ... -e USERID=1001 ...)
ENV USER=triage
ENV USERID=1000

RUN adduser \
        --disabled-password \
        --gecos "" \
        --home "/triage" \
        --no-create-home \
        --uid "${USERID}" \
        "${USER}"         

ENTRYPOINT [ "bash" ]


FROM development AS from-master

LABEL triage.version="master"

RUN pip install git+https://github.com/dssg/triage.git

RUN apk del build-runtime \
    && rm -rf /var/cache/apk/*

ENTRYPOINT [ "triage" ]


FROM development AS production

LABEL triage.version="stable"


RUN pip uninstall triage \
    && pip install triage

RUN apk del build-runtime \
    && rm -rf /var/cache/apk/*

ENTRYPOINT [ "triage" ]
