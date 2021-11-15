function test()
{
    . run/init.sh
    init_environment dlearn test

    local test_id=$1
    shift

    local test_path="src"
    local script_name="test$test_id.py"
    local script_path="$test_path/$script_name"

    python $script_path $@
}

test $@