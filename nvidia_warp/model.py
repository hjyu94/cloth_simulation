import warp as wp
import warp.sim

wp.init()

builder = wp.sim.ModelBuilder()

# anchor point (zero mass)
builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

# build chain
for i in range(1,10):
    builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
    builder.add_spring(i-1, i, 1.e+3, 0.0, 0)

# create model
model = builder.finalize("cuda")

state = model.state()
integrator = wp.sim.SemiImplicitIntegrator()

for i in range(100):

   state.clear_forces()
   integrator.simulate(model, state, state, dt=1.0/60.0)