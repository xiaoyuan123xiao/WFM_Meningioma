# Original WFM Reference

This project is derived from and builds upon:

**WFM: 3D Wavelet Flow Matching for Ultrafast Multi-Modal MRI Synthesis**  
Authors: Yalcin Tur, Mihajlo Stojkovic, and Ulas Bagci  
Conference: Medical Imaging with Deep Learning (MIDL), 2026  
Repository: https://github.com/yalcintur/WFM

The original WFM code is distributed under the MIT License. Its source files are
included in this repository to document provenance and to support the
meningioma-specific reconstruction.

If you use the WFM backbone, please also cite the original WFM paper:

```bibtex
@inproceedings{tur2026wfm,
  title={WFM: 3D Wavelet Flow Matching for Ultrafast Multi-Modal MRI Synthesis},
  author={Tur, Yalcin and Stojkovic, Mihajlo and Bagci, Ulas},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2026}
}
```

This repository adds meningioma T1c preprocessing, DAGCN, high-frequency loss,
anatomical consistency loss, ablation training, registered evaluation, and
article-level statistical analysis.
