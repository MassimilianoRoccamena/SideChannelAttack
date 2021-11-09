function sca()
{
    local environment_id=$1
    local program_name="sca"
    sh run/program.sh $environment_id $program_name
}

sca $@