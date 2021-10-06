# TensorBoard

```bash
# our proxy port, must be > 1024
export PORT=6006
# login to sm-02 with a port forwarding
ssh -D $PORT wilsonb@sm-02.cels.anl.gov
# start tensorboard (load_fast==false is a recent setting that seems to be needed until Tensorflow work's out the bugs)
tensorboard --bind_all --logdir . --load_fast=false
```