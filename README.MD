# PyTorch Interest Points detector

This project implements various different deep learning based interest point methods. Right now only superpoint.


## Requirements

* Python >= 3.5
* PyTorch >= 0.4
* tqdm (Optional for `test.py`)
* tensorboard >= 1.7.0 (Optional for TensorboardX)
* tensorboardX >= 1.2 (Optional for TensorboardX)
* opencv-python

#### Optional
* parmap
* json_minify


## How to use

For overall project structure see the README in source.


### Training MagicPoint

Please modify the configs/magic_point.json. You should set the *data_dir* as that sets where the generated synthetic
 shapes are created. Optionally modify the *save_dir*(where checkpoint files go) and *log dir* (where log files for
 tensorboard go).


Then run

  ```
  python train.py --config configs/magic_point.json
  ```



## References


If using this please cite the originally SuperPoint paper

### BibTeX Citation
```txt
@inproceedings{detone18superpoint,
  author    = {Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {SuperPoint: Self-Supervised Interest Point Detection and Description},
  booktitle = {CVPR Deep Learning for Visual SLAM Workshop},
  year      = {2018},
  url       = {http://arxiv.org/abs/1712.07629}
}
```

The Superpoint portions of this project was ported from <https://github.com/rpautrat/SuperPoint>