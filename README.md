Paper:

Driver Gaze Estimation in the Real World Overcoming the Eyeglass Challenge 

Repository:

https://github.com/arangesh/GPCycleGAN

Steps to run the model:

Clone the repository: https://github.com/arangesh/GPCycleGAN.git

Install all the dependencies given in the Pipfile manually. (all the versions must be compatible to Py 3.8.3):

    pip install numpy
    pip install pillow
    pip install matplotlib
    pip install scipy
    pip install opencv-python
    pip install scikit-learn
    pip install visdom
    pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101
    
The exact dataset used by the project is not available. So I used a subset of the required dataset from this link: https://www.kaggle.com/datasets/yousefradwanlmao/driver-gaze-rgb-dataset-lisa-v1-v2/data

Here, IR camera data is not available. Only RGB.

Prepare the train, val and test splits as follows: 
    python prepare_gaze_data.py --dataset-dir=/path/to/lisat_gaze_data_v2
    
Train the gaze classifier on images without eyeglasses: 
    python gazenet.py --dataset-root-path=/path/to/lisat_gaze_data_v2/rgb_no_glasses/ --version=1_1 --snapshot=./weights/squeezenet1_1_imagenet.pth --random-transforms
    
Train the GPCycleGAN model using the gaze classifier: (Increase the number of epochs as per your GPU)
    python gpcyclegan.py --dataset-root-path=/path/to/lisat_gaze_data_v2/ --data-type=rgb --version=1_1 --snapshot-dir=/path/to/trained/gaze-classifier/directory/ --random-transforms
    
Create fake images using the trained GPCycleGAN model: (Increase the number of epochs, and the batch size as per your GPU)
    python create_fake_images.py --dataset-root-path=/path/to/lisat_gaze_data_v2/rgb_all_data/ --snapshot-dir=/path/to/trained/gpcyclegan/directory/ --version=1_1
    copy /path/to/lisat_gaze_data_v2/rgb_all_data/mean_std.mat /path/to/lisat_gaze_data_v2/rgb_all_data_fake/mean_std.mat # copy over dataset mean/std information to fake data folder
    
Finetune the gaze classifier on all fake images:
    python gazenet-ft.py --dataset-root-path=/path/to/lisat_gaze_data_v2/rgb_all_data_fake/ --version=1_1 --snapshot-dir=/path/to/trained/gpcyclegan/directory/ --random-transforms
    
Inference:
    python infer.py --dataset-root-path=/path/to/lisat_gaze_data_v2/rgb_all_data/ --split=val --version=1_1 --snapshot-dir=/path/to/trained/rgb-models/directory/ --save-viz
    
Please use the updated files that I have attached. The original code has problems. (Only for the attached files)

I have also created a python script to read the .MAT file.
