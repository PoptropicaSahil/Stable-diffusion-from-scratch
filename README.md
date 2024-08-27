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


## Coding CLIP
Looks like an encoder-only model. Check the part on the left for the same. 
![alt text](readme-images/clip-encoder.png)


## Classifier Free Guidance
We inference the model **twice** - one by specifying the prompt and another by not specifying it i.e. unconditioned prompt. Then we combine the output of the model linearly - with a weight `cfg_scale` for how much weight to give to the prompt.
![alt text](readme-images/classifier-free-guidance.png)

## Architecture
The latents along with the prompt embeddings are run through the UNET multiple times. Objective of the UNET is to predict the amount of noise present in the latent, at a given timestamp, for many timestamps. \
Then scheduler will 'remove' noise. \ 
**We should also ensure that while denoising, the outputs remain close to the text prompt.**



![alt text](readme-images/arch-text-to-img.png)

![alt text](readme-images/arch-img-to-img.png)

## Time Embeddings
Similar to how earlier we converted positions to vectors (embeddings), now we will convert timestamps to vectors
![alt text](readme-images/time_embedding.png)
