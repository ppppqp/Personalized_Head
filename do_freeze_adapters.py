import os
import subprocess
def run(layers, comment):
    layers = [str(i) for i in layers]
    args = ["python3", "train_adaptor_snips.py"]
    args.extend(layers)
    output = subprocess.run(args, stdout=subprocess.PIPE).stdout.decode("utf-8")
    eval_output = output.split('\n')[-5:-1]
    with open("final_eval_snips", "a") as f:
        f.write(comment+'\n')
        f.writelines([line + '\n' for line in eval_output])

run([], "all 12 adapters layers")
run(range(0, 6), "last 6 adapters layers")
run(range(6, 12), "first 6 adapters layers")
run(range(1, 12), "first 1 adapters layers")
run(range(0, 11), "last 1 adapters layers")



