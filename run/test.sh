test_program()
{
    local environment_id=$1
    shift
    local test_id=$1
    shift

    . run/init.sh
    init_environment $environment_id test/test$test_id

    local test_path="src"
    local script_name="test$test_id.py"
    local script_path="$test_path/$script_name"

    python $script_path $@
}

test_program $@