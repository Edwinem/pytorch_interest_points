# PyTorch Interest Points detector

This project implements various different deep learning based interest point methods. Right now only superpoint.




## Requirements
* Python >= 3.5
* PyTorch >= 0.4
* tqdm (Optional for `test.py`)
* tensorboard >= 1.7.0 (Optional for TensorboardX)
* tensorboardX >= 1.2 (Optional for TensorboardX)


##References

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