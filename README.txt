required python packages: 

gym ( >= 0.17.3)
numpy (>= 1.19.4)
seaborn ( >= 0.11.0)
Matplotlib ( >= 3.3.3)


To train the model: 
	set parameters at the top of the python script accordingly

episodes: number of training episodes
render_last: True -> show last 20 episodes visually, False -> do not show any 			episodes visually
hyperparameters of Q-learning:
	alpha: corresponds to the learning rate
	gamma: gamma factor
	epsilon: used in e-greedy strategy
hyperparameters of state discretisation:
	B_velocity: number of intervals used to partition the velocity space
	B_position: number of intervals used to partition the position space

