This is the implementation code of the paper "Conditional Video Summarization via Non-monotone
Submodular Function Maximization".

## Get started:
1. System requirement: Pytorch 1.1.0, Python 3.7
2. Download the datasets from the [Google drive link](https://drive.google.com/file/d/1k-3LByZ88Dpx3GGxQhhPkD2PV3aaBfss/view?usp=sharing) and place all datasets under the "datasets" folder. It is about 1 GB and will take some time to finish.  

## Generic Video Summarization:
### Use OVP dataset
1) GoogleNet features  
python QVSmain.py -d datasets/qvs_dataset_ovp_google_pool5.h5 -s datasets/ovpsplitsfull.json -m OVP --mode 2  
2) Histgram color features  
python QVSmain.py -d datasets/qvs_dataset_ovp_color.h5 -s datasets/ovpsplitscolor.json -m OVP --mode 2 --ftype color  

### Use Youtube dataset
1) GoogleNet features  
python QVSmain.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsfull.json -m Youtube --mode 2  
2) Histgram color features  
python QVSmain.py -d datasets/qvs_dataset_youtube_color.h5 -s datasets/youtubesplitscolor.json -m Youtube --mode 2 --ftype color  

## Conditional Video Summarization:  
Using Youtube v73 as an example  
1) Focus on the game field  
python QVSmain.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsv73.json -m Youtube --mode 2 --cond --query 2  
2) Focus on the stands  
python QVSmain.py -d datasets/qvs_dataset_youtube_google_pool5.h5 -s datasets/youtubesplitsv73.json -m Youtube --mode 2 --cond --query 9

## More qualitative results for conditional video summarization:
experiment/OVP21.png
![OVP21](https://github.com/amyxwgu/CVS/blob/main/experiment/OVP21.png)
