# My README file

## TensorBoard

```bash
# our proxy port, must be > 1024
export PORT=6006
# login to sm-02 with a port forwarding
ssh -D $PORT wilsonb@sm-02.cels.anl.gov
# start tensorboard (load_fast==false is a recent setting that seems to be needed until Tensorflow work's out the bugs)
tensorboard --bind_all --logdir . --load_fast=false
```

## Restarting

```bash
python3 train.py --environment AT --architecture FCNetwork --num_actors 2 --fixed_temperatures 1.0 0.5 --td_steps 10 --discount 1 --stored_before_train 20000 --group_tag my_group_tag_random --run_tag my_run_tag_002 --load_state 'runs/AT/my_group_tag_random/my_run_tag_002/saves/359000'
```