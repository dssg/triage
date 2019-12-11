#!/usr/bin/env bash

source .aws_env

# Exit the script as soon as something fails (-e) or if a variable is not defined (-u)
set -e -u

function info () {
	echo "##############################################"
	echo "#                                            #"
	echo "#     Project: ${PROJECT_NAME}                    #"
    echo "#     Triage ver. ${TRIAGE_VERSION}                      #"
	echo "#                                            #"
	echo "##############################################"
	echo "Environment: ${ENV}"
	echo "ECR Registry: ${AWS_REGISTRY}"
	echo "BATCH JOB QUEUE: ${AWS_JOB_QUEUE}"
    echo "DB: ${POSTGRES_DB}"
	echo "S3 Bucket: ${S3_BUCKET}"
	python --version
	pyenv --version
	pip --version
}

function sync_to_s3 () {

    echo "##############################################"
	echo "#                                            #"
	echo "#  Uploading changes to s3://${S3_BUCKET} "
    echo "#                                            #"
	echo "##############################################"

	aws s3 sync triage/experiments/ s3://${S3_BUCKET}/experiments
}

function sync_from_s3 () {

    echo "##############################################"
	echo "#                                            #"
	echo "#  Getting changes from s3://${S3_BUCKET} "
    echo "#                                            #"
	echo "##############################################"

	aws s3 sync s3://${S3_BUCKET}/experiments/ triage/experiments/
}

function update_jobs () {
	echo "Updating the job definition of the following tasks: ${PROJECT_NAME}"

	echo "+----------------------------------------+"
	echo "|                                        |"
	echo "| Updating  ${PROJECT_NAME} job definition"
	echo "|                                        |"
	echo "+----------------------------------------+"

	aws batch register-job-definition --cli-input-json file://infrastructure/aws_batch/triage-job-definition.json
}

function update_images () {
    echo "Updating images related to this project"

    tasks=triage

	echo "Updating the image of the following tasks: ${tasks}"

	for task in ${tasks}
	do
		echo "+----------------------------------------+"
		echo "|                                        |"
		echo "| Updating ${task} image"
		echo "|                                        |"
		echo "+----------------------------------------+"
		docker build --no-cache --tag dsapp/${PROJECT_NAME}/${task} infrastructure/${task}
        docker tag dsapp/${PROJECT_NAME}/${task} ${AWS_REGISTRY}/dsapp/${PROJECT_NAME}/${task}:${TRIAGE_VERSION}
		docker tag dsapp/${PROJECT_NAME}/${task} ${AWS_REGISTRY}/dsapp/${PROJECT_NAME}/${task}:latest

		eval "$(aws ecr get-login --no-include-email --region us-west-2)"

		docker push "${AWS_REGISTRY}"/dsapp/"${PROJECT_NAME}"/${task}:"${TRIAGE_VERSION}"
		docker push "${AWS_REGISTRY}"/dsapp/"${PROJECT_NAME}"/${task}:latest
	done

}

function update_triage_cli_image () {
    tasks=triage-cli

	echo "Updating the image of the following tasks: ${tasks}"

	for task in ${tasks}
	do
		echo "+----------------------------------------+"
		echo "|                                        |"
		echo "| Updating ${task} image"
		echo "|                                        |"
		echo "+----------------------------------------+"
		docker build --no-cache --tag dsapp/${task} infrastructure/triage
        docker tag dsapp/${task} ${AWS_REGISTRY}/dsapp/${task}:${TRIAGE_VERSION}
		docker tag dsapp/${task} ${AWS_REGISTRY}/dsapp/${task}:latest

		eval "$(aws ecr get-login --no-include-email --region us-west-2)"

		docker push "${AWS_REGISTRY}"/dsapp/${task}:"${TRIAGE_VERSION}"
		docker push "${AWS_REGISTRY}"/dsapp/${task}:latest
	done

}



function run_experiment () {
    job_name=$1
    echo "Running job ${job_name}"

	environment_overrides=$2
	echo "Using environment_overrides: ${environment_overrides}"

    parameters=$3
    echo "Using parameters: ${parameters}"

    command_overrides=${@:4}

    # # Retrieve temporary session credentials for current user
    session=$(aws sts get-session-token --duration-seconds 129600)  # 36 h

    # # Restructure these to mirror pipeline overrides
    creds=$(<<<"$session" jq -f infrastructure/aws_batch/credentials.filter)


    # # Merge these AWS session credentials into *all* pipeline overrides
    overrides=$(
        < ${environment_overrides} \
        jq --arg creds "$creds" \
        '.environment += ($creds|fromjson|.environment)'
    )

    if [ ! -z "$command_overrides" ]
	then

		echo "Adding ${command_overrides} to the command"

		for cmd in ${command_overrides}
		do
			overrides=$(echo $overrides | jq --arg cmds "${cmd}" \
											 '.command |= .+ [$cmds]')
		done

	fi

    aws batch submit-job --job-queue ${AWS_JOB_QUEUE} \
		--job-name ${job_name} \
        --job-definition triage-cli-experiment \
        --container-overrides "${overrides}" \
        --parameters "${parameters}"
}

function run() {
    run_experiment $1 infrastructure/aws_batch/triage-overrides.json "${@:2}"
}


function help_menu () {
cat << EOF
Usage: ${0} (-h | -i | -u | -b | -r | -a | --sync_{to,from}_s3 )
OPTIONS:
   -h|--help                   Show this message
   -i|--info                   Show information about the environment
   -t|--update-triage-image    Build the ${PROJECT_NAME}'s triage image and push it to the AWS ECR
   -u|--update-jobs            Update the ${PROJECT_NAME}'s triage job definition in AWS Batch
   -r|--run-experiment         Run experiments on ${PROJECT_NAME} data
   --sync-to-s3                Uploads the experiments and configuration files to ${S3_BUCKET}
   --sync-from-s3              Gets the experiments and configuration files from ${S3_BUCKET}
EXAMPLES:
   Build and push the images to your AWS ECR:
        $ ./deploy.sh -b
   Update the job's definitions:
        $ ./deploy.sh -u
   Sync your experiment config files:
        $ ./deploy.sh --sync-to-s3
   Run triage experiments:
        $ ./deploy.sh -r --experiment_file=s3://${S3_BUCKET}/experiments/test.yaml,output_path=s3://${S3_BUCKET}/triage,replace=--replace

EOF
}

if [[ $# -eq 0 ]] ; then
	help_menu
	exit 0
fi

# Deal with command line flags.
case "${1}" in
  -b|--update-images)
  update_images
  shift
  ;;
  -t|--update-triage-image)
  update_triage_cli_image
  shift
  ;;
  -u|--update-jobs)
  update_jobs
  shift
  ;;
  -r|--run-experiment)
  run ${@:2}
  shift
  ;;
  -a|--all)
  all
  shift
  ;;
  -i|--info)
  info
  shift
  ;;
  --sync-from-s3)
  sync_from_s3
  shift
  ;;
  --sync-to-s3)
  sync_to_s3
  shift
  ;;
  -h|--help)
  help_menu
  shift
  ;;
  *)
  echo "${1} is not a valid flag, try running: ${0} --help"
  ;;
esac
shift
