for i in *.batch; do
    echo $i && sbatch $i && mv $i old/
done
