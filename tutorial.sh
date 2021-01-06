#!/bin/bash

set -e -u

DIRTYDUCK_HOME="dirtyduck"

function help_menu () {
cat << EOF
Usage: ${0} {up|down|build|rebuild|run|logs|status|clean}

OPTIONS:
   -h|help             Show this message
   up                  Starts Food DB
   down                Stops Food DB
   build               Builds images (food_db and bastion)
   rebuild             Builds images (food_db and bastion) ignoring if they already exists
   -l|logs             Shows container's logs
   status              Shows status of the containers
   -d|clean            Removes containers, images, volumes, networks

INFRASTRUCTURE:
   Build the Dirtyduck's DB:
        $ ./tutorial.sh up

   Check the status of the containers:
        $ ./tutorial.sh status

   Stop the dirtyduck's infrastructure:
        $ ./tutorial.sh down

   Destroy all the resources related to the tutorial:
        $ ./tutorial.sh clean

   View dirtyduck's logs:
        $ ./tutorial.sh -l

EOF
}

function start_infrastructure () {
    docker-compose -f ${DIRTYDUCK_HOME}/docker-compose.yml  up -d food_db
}

function stop_infrastructure () {
    docker-compose -f ${DIRTYDUCK_HOME}/docker-compose.yml  stop
}

function build_triage () {
     docker build --target development -t dsapp/triage:development -f Dockerfile .
}

function build_images () {
     docker-compose -f ${DIRTYDUCK_HOME}/docker-compose.yml build "${@}"
}

function destroy () {
     docker-compose -f ${DIRTYDUCK_HOME}/docker-compose.yml  down --rmi all --remove-orphans --volumes
}

function infrastructure_logs () {
    docker-compose -f ${DIRTYDUCK_HOME}/docker-compose.yml logs -f -t
}

function status () {
	docker-compose -f ${DIRTYDUCK_HOME}/docker-compose.yml ps
}

function bastion () {
    docker-compose -f ${DIRTYDUCK_HOME}/docker-compose.yml run --service-ports  --rm --name dirtyduck_bastion bastion
}

function all () {
	build_images
	start_infrastructure
	status
}


if [[ $# -eq 0 ]] ; then
	help_menu
	exit 0
fi

case "$1" in
    up)
        start_infrastructure
		shift
        ;;
    down)
        stop_infrastructure
		shift
        ;;
    build)
        build_triage
        build_images
		shift
        ;;
    rebuild)
        build_images --no-cache
		shift
        ;;
    -d|clean)
        destroy
		shift
        ;;
    -l|logs)
        infrastructure_logs
		shift
        ;;
    status)
        status
		shift
        ;;
    bastion)
        bastion
	        shift
	;;
    -h|help)
        help_menu
                shift
        ;;
   *)
       echo "${1} is not a valid flag, try running: ${0} -h"
	   shift
       ;;
esac
shift

cd - > /dev/null
