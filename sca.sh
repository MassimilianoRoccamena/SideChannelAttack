function sca()
{
    local environment_id=$1
    local program_name="sca"
    sh run/aidenv.sh $environment_id $program_name
}

sca $@