for MODEL in BERT
do
    for HEAD in 8 4 2
    do
        for DIM in 512 1024 2048
        do
            python train_clinc_plus.py -m ${MODEL} -dim ${DIM} -nh ${HEAD}
        done
    done
done
