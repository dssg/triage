FROM python:3.7-slim AS development

LABEL creator="Center for Data Science and Public Policy (DSaPP)" \
        maintainer="Adolfo De Unánue <adolfo@cmu.edu>" \
      triage.version="development"


RUN apt-get update && \
        apt-get install -y --no-install-recommends gcc build-essential libpq-dev liblapack-dev postgresql git

RUN apt-get update -y && \
        apt-get install -y --no-install-recommends gnupg2 wget && \
        wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
        echo "deb http://apt.postgresql.org/pub/repos/apt/ buster-pgdg main" | tee  /etc/apt/sources.list.d/pgdg.list && \
        apt-get update -y && \
        apt-get install -y --no-install-recommends postgresql-client-12

RUN mkdir -p triage

WORKDIR triage

ENV SHELL=/bin/bash
ENV USERNAME=triage
ENV USERID=1000
ENV TRIAGE_IMAGE=development

RUN adduser \
        --disabled-password \
        --gecos "" \
        --home "/home/triage" \
        --uid "${USERID}" \
        "${USERNAME}"


RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]triage@$(tput setaf 5)${TRIAGE_IMAGE}$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' >> /home/triage/.bashrc

RUN mkdir -p /opt/venv
RUN chown -R triage:triage /opt/venv
RUN chown -R triage:triage .

USER ${USERNAME}

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=triage:triage requirement/ requirement/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirement/main.txt && \
    pip install --no-cache-dir -r requirement/test.txt && \
    pip install --no-cache-dir -r requirement/extras-rq.txt && \
    pip install --no-cache-dir ipython jupyter

COPY --chown=triage:triage README.md .
COPY --chown=triage:triage LICENSE .
COPY --chown=triage:triage src/ src/
COPY --chown=triage:triage setup.py .

RUN pip install -e .

ENTRYPOINT [ "bash" ]

FROM python:3.7-slim AS master

LABEL triage.version="master"

COPY --from=development /opt/venv /opt/venv

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
        apt-get install -y --no-install-recommends git libpq-dev

RUN pip install --no-cache-dir git+https://github.com/dssg/triage@master

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

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]triage@$(tput setaf 6)master$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' > /home/triage/.bashrc

RUN chown -R triage:triage .

USER ${USERNAME}

ENTRYPOINT [ "triage" ]


FROM master AS production

LABEL triage.version="production"

COPY --from=development /opt/venv /opt/venv

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

RUN pip uninstall triage

RUN pip install --no-cache-dir triage

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]triage@$(tput setaf 6)production$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' > /home/triage/.bashrc

RUN chown -R triage:triage .

USER ${USERNAME}

ENTRYPOINT [ "triage" ]
