CONFIG_PATH="config"
AIDENV_PATH="$CONFIG_PATH/aidenv"

ENVIRONMENT_CONFIG_EXT="conf"
PROGRAM_CONFIG_EXT="yaml"

DATADIR_PATH="$AIDENV_PATH/datadir"

init_input()
{
    local input_file="input"
    local input_path="$DATADIR_PATH/$input_file"
    export AIDENV_INPUT=$(cat "$input_path.$ENVIRONMENT_CONFIG_EXT")
}

init_output()
{
    local output_file="output"
    local output_path="$DATADIR_PATH/$output_file"
    export AIDENV_OUTPUT=$(cat "$output_path.$ENVIRONMENT_CONFIG_EXT")
}

NEPTUNE_PATH="$AIDENV_PATH/neptune"

init_neptune()
{
    local user_file="user"
    local user_path="$NEPTUNE_PATH/$user_file"
    export AIDENV_NEPTUNE_USER=$(cat "$user_path.$ENVIRONMENT_CONFIG_EXT")

    local token_file="token"
    local token_path="$NEPTUNE_PATH/$token_file"
    export AIDENV_NEPTUNE_TOKEN=$(cat "$token_path.$ENVIRONMENT_CONFIG_EXT")

    local project_file="project"
    local project_path="$NEPTUNE_PATH/$project_file"
    export AIDENV_NEPTUNE_PROJECT=$(cat "$project_path.$ENVIRONMENT_CONFIG_EXT")
}

init_program()
{
    local relative_path=$1
    local absolute_path="$CONFIG_PATH/$relative_path"
    export AIDENV_PROGRAM="$absolute_path.$PROGRAM_CONFIG_EXT"
}

init_environment()
{
    init_input
    init_output

    init_neptune

    local program_path=$1
    init_program $program_path
}
