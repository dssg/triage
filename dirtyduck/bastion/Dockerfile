FROM python:3.6-stretch

## Installing clients
RUN  sh -c "echo 'deb http://apt.postgresql.org/pub/repos/apt/ stretch-pgdg main' > /etc/apt/sources.list.d/pgdg.list" && \
     wget --quiet -O - http://apt.postgresql.org/pub/repos/apt/ACCC4CF8.asc | apt-key add - && \
     apt-get -y update && \
     apt-get -y install less gdal-bin \
     libpq-dev postgresql-client-10

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR triage
