CONFIG_PATH="config"

setup_neptune()
{
    local neptune_dir="neptune"
    local neptune_path="$CONFIG_PATH/$neptune_dir"
    local file_ext="conf"

    local user_file="user"
    local user_path="$neptune_path/$user_file"
    export NEPTUNE_USER=$(cat "$user_path.$file_ext")

    local token_file="token"
    local token_path="$neptune_path/$token_file"
    export NEPTUNE_TOKEN=$(cat "$token_path.$file_ext")

    local project_file="user"
    local project_path="$neptune_path/$project_file"
    export NEPTUNE_PROJECT=$(cat "$project_path.$file_ext")
}

setup_config()
{
    local program_name=$2
    local program_path="$CONFIG_PATH/$program_name"
    local environment_id=$1
    local file_ext="yaml"
    local file_name="$environment_id.$file_ext"
    export AIDENV_CONFIG="$program_path/$file_name"
}

setup_program()
{
    setup_neptune
    
    local environment_id=$1
    local program_name=$2
    setup_config $environment_id $program_name
}