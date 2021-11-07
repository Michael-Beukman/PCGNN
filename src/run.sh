export PYTHONPATH=`pwd`/external/gym-pcgrl:`pwd`

if [ -f $1 ]; then
    python -u  "$@"
else
    python -u  ../"$@"
fi

