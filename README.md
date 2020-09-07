# SwAV-TF

By: [Ayush Thakur](https://twitter.com/ayushthakur0) and [Sayak Paul](https://twitter.com/RisingSayak)

TensorFlow 2 implementation of [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882) by Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin.

To know more about SwAV and the experiments we performed with it you can check out our blog post [Unsupervised Visual Representation Learning with SwAV](https://app.wandb.ai/authors/swav-tf/reports/Unsupervised-visual-representation-learning-with-SwAV--VmlldzoyMjg3Mzg). Our blog post is structured in the following way - 

- Common problems in existing self-supervised methods (for visual representation learning)
- SwAV at a high-level
- Multi-crop augmentation policy and its promise
- Contrasting the cluster assignments
  - Cluster assignment as an optimal transport problem
  - Swapped prediction problem to enforce consistency
- A single forward pass in SwAV
- Experimental results
- Conclusion

## Model weights for reproducibility
Available [here](https://github.com/ayulockin/SwAV-TF/releases/tag/v0.1.0). 

## Results
Results of all our experiments are available as a `wandb` dashboard. [![Explore-in W&B](https://img.shields.io/badge/Explore--in-W%26B-%23FFBE00)](https://app.wandb.ai/authors/swav-tf) 

## Acknowledgements
Thanks to Mathilde Caron for providing insightful pointers that helped us minimally implement SwAV.

Thanks to Jiri Simsa of Google for providing us with tips that helped us improve our data input pipeline.

Thanks to the [Google Developers Experts program](https://developers.google.com/programs/experts/0) for providing us with GCP credits.
