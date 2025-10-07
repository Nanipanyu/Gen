# 🔹 EDM Sampling Algorithm

# EDM = Elucidated Diffusion Model, which refines how we sample (generate images) from a trained diffusion model.
# Instead of using naive ancestral sampling (like DDPM), EDM introduces more stable ODE/SDE solvers for the reverse diffusion process.

# 🔹 Second-order solver (EDM sampling)

# The diffusion process is like an ODE:

# dx/dt = f(x,t)

# where 
# 𝑓
# f depends on the denoiser (the trained network).

# To generate samples, we integrate this ODE backwards (from noise to image).

# The second-order solver means:

# Instead of using a simple Euler step (first-order), which updates like


# EDM uses a second-order Heun’s method (a predictor–corrector scheme):

# Predictor step (Euler):

# Corrector step:


# This improves accuracy and stability for the same number of steps.

# 🔹 Why second-order solver?

# First-order solvers (Euler/ancestral) need many steps to avoid artifacts.

# Second-order (Heun’s method) achieves better image quality with fewer steps, because it corrects the drift introduced in each step.

# This is why EDM can generate high-quality images in as few as 18–40 steps, instead of hundreds.