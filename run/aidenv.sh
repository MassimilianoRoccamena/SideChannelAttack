run_environment()
{
    # build args
    local environment_id=$1
    shift

    local main_path="src/main"
    local script_name="$environment_id.py"
    local script_path="$main_path/$script_name"

    local program_args=$@

    # call python
    python $script_path $program_args
}

run_program()
{
    # build args
    local environment_id=$1
    local program_name=$2
    shift

    # init environment
    . run/init.sh
    init_environment $environment_id $program_name

    # call environment
    local program_args=$@
    run_environment $environment_id $program_args
}

run_program $@