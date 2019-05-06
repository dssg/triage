#!/bin/bash

set -e -u

PROJECT="triage-dirtyduck"
PROJECT_HOME="$( cd "$( dirname "$0" )" && pwd )"
INFRASTRUCTURE_HOME="${PROJECT_HOME}/dirtyduck"

cd "$INFRASTRUCTURE_HOME"

function help_menu () {
cat << EOF
Usage: ${0} {start|stop|build|rebuild|run|logs|status|destroy|all|}

OPTIONS:
   -h|help             Show this message
   start               Starts Food DB
   stop                Stops Food DB
   build               Builds images (food_db and bastion)
   rebuild             Builds images (food_db and bastion) ignoring if they already exists
   -l|logs             Shows container's logs
   status              Shows status of the containers
   -d|clean            Removes containers, images, volumes, netrowrks

INFRASTRUCTURE:
   Build the infrastructure:
        $ ./tutorial.sh start

   Check the status of the containers:
        $ ./tutorial.sh status

   Stop the tutorial's infrastructure:
        $ ./tutorial.sh stop

   Destroy all the resources related to the tutorial:
        $ ./tutorial.sh clean

   View the infrastructure logs:
        $ ./tutorial.sh -l

EOF
}

function start_infrastructure () {
    docker-compose up -d food_db
}

function stop_infrastructure () {
	docker-compose  stop
}

function build_images () {
	docker-compose  build "${@}"
}

function destroy () {
	docker-compose  down --rmi all --remove-orphans --volumes
}

function infrastructure_logs () {
    docker-compose logs -f -t
}

function status () {
	docker-compose ps
}

function bastion () {
    docker-compose run --service-ports  --rm --name tutorial_bastion bastion
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
    start)
        start_infrastructure
		shift
        ;;
    stop)
        stop_infrastructure
		shift
        ;;
    build)
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
    -h|--help)
        help_menu
                shift
        ;;
   *)
       echo "${1} is not a valid flag, try running: ${0} --help"
	   shift
       ;;
esac
shift

cd - > /dev/null
