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
    ncurses \
    "

RUN apk add --virtual build-runtime \
     build-base python-dev openblas-dev lapack-dev freetype-dev pkgconfig gfortran g++ gcc git curl  \
     && ln -s /usr/include/locale.h /usr/include/xlocale.h \
     && pip install --upgrade pip setuptools \
     && rm -r /root/.cache \
        && apk add --no-cache --virtual build-dependencies $PACKAGES

RUN mkdir triage

WORKDIR triage

ENV SHELL=/bin/bash
ENV USERNAME=triage
ENV USERID=1000

RUN adduser \
        --disabled-password \
        --gecos "" \
        --home "/home/triage" \
        --uid "${USERID}" \
        "${USERNAME}"         

USER ${USERNAME}

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]triage@$(tput setaf 5)development$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' >> /home/triage/.bashrc

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

ENTRYPOINT [ "bash" ]

FROM development AS from-master

LABEL triage.version="master"

USER root

RUN pip install git+https://github.com/dssg/triage.git

RUN apk del build-runtime \
    && rm -rf /var/cache/apk/*

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]triage@$(tput setaf 6)master$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' > /home/triage/.bashrc

ENTRYPOINT [ "triage" ]

FROM from-master AS dirtyduck

LABEL triage.version="dirtyduck (from master)"

USER root

RUN pip install ipython jupyter

USER ${USERNAME}

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]\u@$(tput setaf 6)$(tput smul)dirtyduck$(tput rmul)$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' > /home/triage/.bashrc

ENTRYPOINT [ "bash" ]

FROM development AS production

LABEL triage.version="stable"

USER root

RUN pip uninstall triage ipython jupyter \
    && pip install triage

RUN apk del build-runtime \
    && rm -rf /var/cache/apk/*


USER ${USERNAME}

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]triage@$(tput setaf 2)stable$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' > /home/triage/.bashrc

ENTRYPOINT [ "triage" ]

