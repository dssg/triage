FROM python:3.6

LABEL triage.version="v3.3.0" \
      triage.from="cli" \
      creator="Center for Data Science and Public Policy (DSaPP)" \
      maintainer="Adolfo De Unánue <adolfo@uchicago.edu>"

RUN apt update

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir triage

WORKDIR triage

ENTRYPOINT [ "triage", "experiment" ]
