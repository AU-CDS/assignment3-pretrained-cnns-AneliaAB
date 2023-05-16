[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Aj7Sf-j_)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11016919&assignment_repo_type=AssignmentRepo)
# Using pretrained CNNs for image classification

## Project description by Ross
In the previous assignments involving classification, we were performing a kind of simple feature extraction on images by making them greyscale and flattening them to a single vector. This vector of pixel values was then used as the input for some kind of classification model.

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which trains a classifier on this dataset using a *pretrained CNN like VGG16*
- Save the training and validation history plots
- Save the classification report

Tips

- You should not upload the data to your repo - it's around 3GB in size.
  - Instead, you should document in the README file where your data comes from, how a user should find it, and where it should be saved in order for your code to work correctly.
- The data comes already split into training, test, and validation datasets. You can use these in a ```TensorFlow``` data generator pipeline like we saw in class this week - you can see an example of that [here](https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator).
- There are a lot of images, around 106k in total. Make sure to reserve enough time for running your code!
- The image labels are in the metadata folder, stored as JSON files. These can be read into ```pandas``` using ```read_json()```. You can find the documentation for that online.

## Data

This project trains a classifier on the IndoFashion dataset using VGG. The data consists of 106K images and 15 unique cloth categories. The data is split into training, test and validation and can be accessed on Kaggle via this link: [https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset)

The dataset is not is the repository and needs to be loaded separately.

## How to Install and Run the Project

Loading the data:

1. Download the data via the link above. 
2. Load the data outside of this repository. Your working repository should look like this:

├── current directory/project repository

├── images

|   └── metadata

|   └── test

|   ├── train

|   ├── val

Installation:

1. First you need to clone the project repository
2. Navigate from the root of your working directory to **assignment3-pretrained-cnns-AneliaAB**
3. Run the setup file, which will install all the requirements by writing `bash setup.sh` in the terminal
4. Navigate to the folder `scr` and run the script by writing `python code.py` in the terminal or run the notebook `code.ipynb` 

## Results

After installing and running the scripts, you should be able to see the results in the `out` folder. 

- Classification report
- Training and validation history plots