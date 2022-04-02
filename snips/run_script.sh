for MODEL in BERT
do
    for epoch in 25 50 75 100
    do
        for DIM in 16 32 64 128
        do
            for HEAD in 2 4 8
            do
                python train_snips_squezed.py -m ${MODEL} -dim ${DIM} -nh ${HEAD} -e ${epoch}
            done
        done
   done
done