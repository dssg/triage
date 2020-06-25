FROM python:3.7-slim as development

LABEL creator="Center for Data Science and Public Policy (DSaPP)" \
        maintainer="Adolfo De Unánue <adolfo@cmu.edu>" \
      triage.version="development"


RUN apt-get update && \
        apt-get install -y --no-install-recommends gcc build-essential libpq-dev liblapack-dev postgresql


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


RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]triage@$(tput setaf 5)development$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' >> /home/triage/.bashrc


COPY README.md .
COPY LICENSE .


RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirement/ requirement/
RUN pip install -r requirement/main.txt
RUN pip install -r requirement/test.txt

COPY src/ src/
COPY setup.py .

RUN pip install .

RUN chown -R triage:triage .

USER ${USERNAME}

ENTRYPOINT [ "bash" ]

FROM python:3.7-slim AS master

LABEL triage.version="master"

COPY --from=development /opt/venv /opt/venv

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

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
