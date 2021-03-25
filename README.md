# EpisodicControllers


### Model Free Episodic Controller [done]

### Neural Episodic Controller [arXiv](https://arxiv.org/abs/1703.01988v1) [in-progress]

#### Components
- A Differentiable Neural Dictionary (DND) for each action
- A network that converts output of DND lookup to Q values

##### DND
- Each action has a seperate DND
- Lookup steps
   - Lookup K-NN Q value - is K 1?
   - Take action based on the Q value and epsilon like always
   -  Add this new Q value for this state-action to the DND (update if already present using Q value update[1])
   - When memory is max, overwrite new state-action as the nearest key

[1]. Q <- Q + alpha*(Q_new - Q)  
#### How does it work?

