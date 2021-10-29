script_id=$1
shift

root_path="script"
file_name="run-$script_id.py"
file_path="$root_path/$file_name"

python $file_path $@