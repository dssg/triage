# Triage database provisioner

This document explains the purpose and behavior of the Triage database provisioner, accessed from the Triage CLI. It is optional and only intended for use if you don't have an existing Postgres database to use for Triage.

The Triage database provisioner is just a single command:

`triage db up`

This command attempts to use docker to spawn a new Postgres 12 database. If successful, it will prompt you for a password to use for a user, and populate the connection information in database.yaml. The next time you run `triage db up`, it will look for the existing container and reuse it.

## Troubleshooting

### No docker
The command does require some version of Docker. We recommend getting it from the [official Docker downloads page](https://docs.docker.com/get-docker/).

### Can't log in
Because of the way Docker volumes work, if you manually remove the Docker container created by `triage db up`, the volume will still be around. This is usually fine, but the superuser credential information will persist as well, which means the next time you spawn the database, *the Postgres server will not take the new credential information into account*. Under normal usage (simply calling `triage db up` and never removing the container), you will never run into this situation. But if you do, and you would like to use a new username/password, you will have to remove the volume before recreating. This can be done with `docker volume rm db-data`. This will also remove all of the stored data in Postgres, so beware!
