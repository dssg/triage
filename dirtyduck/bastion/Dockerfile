FROM dsapp/triage:master

RUN echo 'export PS1="\[$(tput setaf 4)$(tput bold)[\]\u@$(tput setaf 2)$(tput smul)dirtyduck$(tput rmul)$(tput setaf 4)$:\\w]#\[$(tput sgr0) \]"' > /home/triage/.bashrc

USER root

RUN apt-get update -y && \
        apt-get install -y --no-install-recommends gnupg2 wget && \
        wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
        echo "deb http://apt.postgresql.org/pub/repos/apt/ buster-pgdg main" | tee  /etc/apt/sources.list.d/pgdg.list && \
        apt-get update -y && \
        apt-get install -y --no-install-recommends postgresql-client-12

RUN pip install ipython jupyter


USER triage

ENTRYPOINT [ "bash" ]
