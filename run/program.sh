function aidenv_program()
{
    # build args
    local environment_id=$1
    local program_name=$2
    shift

    # configure program
    . run/setup.sh
    setup_program $environment_id $program_name

    # call aidenv
    local program_args=$@
    sh run/environment.sh $environment_id $program_args
}

aidenv_program $@