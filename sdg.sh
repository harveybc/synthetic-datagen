export PREV_PYTHONPATH=$PYTHONPATH
export PYTHONPATH=./:$PYTHONPATH

echo "--- Debug from sdg.sh ---"
echo "Which python: $(which python)"
echo "Python version:"
python --version
echo "PYTHONPATH: $PYTHONPATH"
echo "--- End Debug ---"

python app/main.py "$@"

export PYTHONPATH=$PREV_PYTHONPATH