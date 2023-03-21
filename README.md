## Distilling Dynamics
  
Peter Whidden, compute(r)ender

--- 

Models trained on large datasets often capture information which can be applied beyond their originally intended tasks and domains. [Dream Fields](https://arxiv.org/abs/2112.01455) showed that 3D structure can be extracted from a CLIP model trained on only 2D images by distilling it into a NeRF model that enforces consistency in 3D space. [DreamFusion](https://arxiv.org/abs/2209.14988) improved upon these results using a diffusion model. This approach is intuitive, given that these models are capable of representing a particular subject from multiple perspectives. Additionally, however, these models may also be able to represent a subject at multiple points in time. Given this, maybe it is possible that dynamics could be distilled into a model which enforces a consistent structure over time and space. This code explores some simple ways that this might be realized.

For example, here is Stable Diffusion conditioned with the prompt "raindrops on glass" distilled into a [Neural Cellular Automata](https://distill.pub/selforg/2021/textures/) model:  

![nca raindrops](assets/raindrops.gif)
![nca raindrops 2](assets/raindrops-2-scaled-fast.gif)

The dynamics in this experiment likely owe more to the strong inductive bias of NCA than to dynamics which may exist in the diffusion model. Currently, this method is mainly aspirational.



