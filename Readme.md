This is the implementation codes for paper "Conditional Video Summarization via Non-monotone
Submodular Function Maximization".

## Get started:
1. System requirement: Pytorch 1.1.0, Python 3.7
2. (Optional) Download the datasets from the [Google drive link](https://drive.google.com/file/d/1GBku5FGII9KHX47f5rW2iRx6wyaNp9E2/view?usp=sharing) and place all datasets under the "[datasets](/datasets)" folder. GoogleNet testing features are available in the current "datasets" folder.

## Generic Video Summarization:
### Use OVP dataset
1) GoogleNet features (can be run without downloading additional data)  
python QVSmain_gh.py -d datasets/qvs_dataset_ovp_google_pool5.h5 -s datasets/ovpsplitsfull.json -m OVP --mode 2
2) Histgram color features  
python QVSmain_gh.py -d datasets/qvs_dataset_ovp_color.h5 -s datasets/ovpsplitscolor.json -m OVP --mode 2 --ftype color

### Use Youtube dataset
1) GoogleNet features (can be run without downloading additional data)  
python QVSmain_gh.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsfull.json -m Youtube --mode 2
2) Histgram color features  
python QVSmain_gh.py -d datasets/qvs_dataset_youtube_color.h5 -s datasets/youtubesplitscolor.json -m Youtube --mode 2 --ftype color

## Conditional Video Summarization:
Using Youtube v73 as an example (can be run without downloading additional data):  
1) Focus on the game field  
python QVSmain_gh.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsv73.json -m Youtube --mode 2 --cond --query 2
2) Focus on the stands  
python QVSmain_gh.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsv73.json -m Youtube --mode 2 --cond --query 9
