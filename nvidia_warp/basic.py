import os
import math

import numpy as np

import torch

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    # seconds
    sim_duration = 2.0

    # control frequency
    frame_dt = 1.0 / 60.0
    frame_steps = int(sim_duration / frame_dt) # 120 frames

    # sim frequency
    sim_substeps = 16
    sim_steps = frame_steps * sim_substeps # 120 * 16 steps
    sim_dt = frame_dt / sim_substeps # 1 step 을 지나는데 걸리는 시간

    render_time = 0.0

    train_iters = 3
    train_rate = 0.01

    # ke = 1.0e4
    # kf = 0.0
    # kd = 1.0e1
    # mu = 0.25


    def __init__(self, stage):
        self.create_scene(stage)


    def create_scene(self, stage):
        builder = wp.sim.ModelBuilder()

        self.sim_width = 4
        self.sim_height = 4

        builder.add_cloth_grid(
            pos=(0.0, 4.0, 0.0),
            rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.5),
            vel=(0.0, 0.0, 0.0),
            dim_x=self.sim_width,
            dim_y=self.sim_height,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_left=True,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e1,
        )
        
        self.model = builder.finalize(device="cuda", requires_grad=True)
        self.model.ground = True

        # self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=1.0)
        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=40.0)
        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = wp.vec3(0.0, 0.0, 0.0)
        self.com = wp.zeros(1, dtype=wp.vec3, device="cuda", requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, device="cuda", requires_grad=True)

        self.states = []
        for i in range(self.sim_steps + 1):
            self.states.append(self.model.state(requires_grad=True))
        
        wp.sim.collide(self.model, self.states[0])


    @wp.kernel
    def com_kernel(positions: wp.array(dtype=wp.vec3), n: int, com: wp.array(dtype=wp.vec3)):
        tid = wp.tid()

        # compute center of mass
        wp.atomic_add(com, 0, positions[tid] / float(n))


    @wp.kernel
    def loss_kernel(com: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
        # sq. distance to target
        delta = com[0] - target

        loss[0] = wp.dot(delta, delta)


    @wp.kernel
    def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
        tid = wp.tid()

        # gradient descent step
        x[tid] = x[tid] - grad[tid] * alpha


    def render(self, iter):
        for i in range(0, self.sim_steps, self.sim_substeps):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.states[i])
            self.renderer.end_frame()
        
            self.render_time += self.frame_dt


    def train_graph(self):
        ##### start creating graph #####
        wp.capture_begin()

        tape = wp.Tape()

        # forward simulation
        with tape:
            # run control loop
            for i in range(self.sim_steps):
                self.states[i].clear_forces()
                self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

            # compute loss on final state
            self.com.zero_()
            wp.launch(
                self.com_kernel,
                dim=self.model.particle_count,
                inputs=[self.states[-1].particle_q, self.model.particle_count, self.com],
                device="cuda",
            )

            wp.launch(
                kernel=self.loss_kernel,
                dim=1, 
                inputs=[self.com, self.target, self.loss], 
                device="cuda"
            )

        # backwoard simulation
        tape.backward(self.loss)

        self.graph = wp.capture_end()
        ##### end creating graph #####

        # replay and optimize
        for i in range(self.train_iters):
            with wp.ScopedTimer("Step"):
                # forward + backward
                wp.capture_launch(self.graph)

                # gradient descent step
                x = self.states[0].particle_q
                wp.launch(
                    kernel=self.step_kernel, 
                    dim=len(x), 
                    inputs=[x, x.grad, self.train_rate], 
                    device="cuda"
                )
                
                # debug
                print(f"Iter: {i} Loss: {self.loss}")

                # clear grads for next iteration
                tape.zero()
            
            with wp.ScopedTimer("Render"):
                self.render(i)


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/basic.usd")
    example = Example(stage_path)
    example.train_graph()
    example.renderer.save()