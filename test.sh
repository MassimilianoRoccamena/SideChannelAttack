function test()
{
    local test_path="src/test"
    local script_name="test-main.py"
    local script_path="$test_path/$script_name"

    python $script_path $@
}

test $@