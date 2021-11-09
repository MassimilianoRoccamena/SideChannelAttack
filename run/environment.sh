function aidenv_environment()
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

aidenv_environment $@