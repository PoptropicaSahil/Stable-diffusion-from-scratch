# Coding Stable Diffusion from Scratch

> Like my previous implementation, this repo is a code follow-along excercise by the [excellent tutorial](https://www.youtube.com/watch?v=ZBKpAp_6TGI) by Umar Jamil `@hkproj`. He read more than 30 papers for the 5-hour video tutorial, which is already incredible. 

The goal of this repo is to 
- Increase my and the reader's understanding Diffusion models 
- Develop a visual representation as a flowchart of the entire repository. The previous flowchart on LLaMA2 gathered a lot of encouragement, so why not!


## What is a generative model
A generative model learns a probability distribution of the data set such that we can then sample from the distribution to create new instances of data.

![alt text](readme-images/data-distribution.png)


## Background
![alt text](readme-images/background.png)
![alt text](readme-images/training-sampling.png)
![alt text](readme-images/overview.png)
![alt text](readme-images/classifier-guidance.png)


## Group Normalization
![alt text](readme-images/layer-norm.png) \
Idea of using group normalization is that because the matrices are subject to pooling, items closer are more likely to be related. So we are normalizing by the groups. <br>
But main idea ofcourse to not let the loss function oscillate too much --> **Training faster**. 

![alt text](readme-images/group-norm.png)


## Need for attention 
Taken from Jay Alammar's blog. The need for attention is to ensure model understands both text and images' embeddings together and can pay *attention* to similar items in that space.
![alt text](readme-images/jalammar-attention.jpeg)
