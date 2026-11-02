# CSC3067 Video Analytics and Machine Learning · Face Detection

Queen's University Belfast · Year 3 · Group project (G19)

A face detection pipeline in **MATLAB** that compares different feature extractors and classifiers. It finds faces in full images using a sliding window followed by non-maximum suppression.

## Pipeline

1. **Features**: raw pixels, HOG (`hog_feature_vector.m`), Gabor (`gabor_feature_vector.m`), and PCA dimensionality reduction (`PrincipalComponentAnalysis.m`)
2. **Classifiers**: nearest neighbour (`NNtraining.m` / `NNTesting.m`), k-NN (`KNNtraining.m` / `KNNTesting.m`), and SVM (using the bundled `SVM-KM` toolbox)
3. **Detection**: sliding window over full images (`slidingWindow_*.m`), then non-maximum suppression (`applyNMS.m`)
4. **Evaluation**: comparison against labelled ground-truth boxes (`ground_*.m`, `ground_box.mat`)

## Project structure

| Path | What's in it |
|---|---|
| `code+images/` | All MATLAB code, datasets (`*.cdataset`) and test images |
| `code+images/images/` | Training and test face images |
| `code+images/LabelingProject/` | Ground-truth labelling project |
| `code+images/SVM-KM/` | Third-party SVM toolbox |

## Running

Open `code+images/` in MATLAB and run `assembled.m` for the full pipeline. To try a single feature and classifier combination, run one of the `slidingWindow_<feature>_<classifier>.m` scripts.

> Note: this repo has some very long file paths. On Windows, clone it with `git -c core.longpaths=true clone ...`.
