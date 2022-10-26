This is the implementation codes for paper "Conditional Video Summarization via Non-monotone
Submodular Function Maximization".

## Get started:
1. System requirement: Pytorch 1.1.0, Python 3.7
2. Download the datasets from the Google drive link.

## Generic Video Summarization:
1. Use OVP dataset
1) GoogleNet features
python QVSmain.py -d datasets/qvs_dataset_ovp_google_pool5.h5 -s datasets/ovpsplitsfull.json -m OVP --mode 2
2) Histgram color features
python QVSmain.py -d datasets/qvs_dataset_ovp_color.h5 -s datasets/ovpsplitscolor.json -m OVP --mode 2 --ftype color

2. Use Youtube dataset
1) GoogleNet features
python QVSmain.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsfull.json -m Youtube --mode 2
2) Histgram color features
python QVSmain.py -d datasets/qvs_dataset_youtube_color.h5 -s datasets/youtubesplitscolor.json -m Youtube --mode 2 --ftype color

## Conditional Video Summarization:
Using Youtube v73 as an example:
1) Focus on the game field
python QVSmain.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsv73.json -m Youtube --mode 2 --cond --query 2
2) Focus on the stands
python QVSmain.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsv73.json -m Youtube --mode 2 --cond --query 9
