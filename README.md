# flatiron_phase4_project
![stjstretch](./img/stjstretch.JPG)

Classifying images of lung X-rays using a Convolutional Neural Network.


Presented by: Rachel Edwards, Svitlana Glibova, Jamie Dowat

## Overview
The purpose of this project is to use an iterative machine learning process to identify pnuemonia in pediatric x-rays. This process involves building convolutional nueral nets and is broken down into a a binary classification process of pnuemonia or not pnuemonia and a ternary classifier determining, bacterial pnuemonia, viral pnuemoina or not pnuemonia. We hope that with the results of this project we will be better able to serve the over worked medical community during the Covid-19 as well as the children we hope to give a more confident diagnoeses too.


## Business Problem
Pnuemonia is one of the leading causes of death in children under five. It is estimated that there are 120 million cases of pnuemonia annually worldwide which results in almost 1.3 million deaths. Pnuemonia has become a growing concern in 2020 as it is a symptom caused by a severe case of covid-19. Children make up 11% of covid cases in the US. It is our goal to develop a tool that is better able to detect pediatric viral pnuemonia. We aim to speicfially target a model that produces the least amount of false negatives and to focus on the recall.

## Data
Our data consists of chest x-ray photos of pediatric patients of one to fives years old from Guangzhou Women and Children's Medical Center, Guangzhou. The data is split into 2 main folders, test and train and then further split into NORMAL or PNUEMONIA. This data in its unprocessed form can be found here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia 

## Methods
Our methods involve creating two convolutional nueral networks, one for binary classification and one for ternary classification. We go through an iterative model process where the model is tweaked after each cylce to perform better than the last. Our target is reducing false negatives and we evaluate that success based on our recall, loss and accuracy metrics. 


## Results


## Conclusions
Final Model

        
 

### Next Steps


## For More Information

## Repository Structure
```
├── img
├── notebooks
├── src
  ├── .py
  ├──.yml
├── README.md
└── final_notebook.ipynb
