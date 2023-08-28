import os
import math

import numpy as np

import torch

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    sim_duration = 0.6
        
    # frame_dt = 1.0 / 60.0                       # 1 frame: 1/60 seconds
    # frame_steps = int(sim_duration / frame_dt)  # 36 frames

    # sim_substeps = 32
    # sim_steps = frame_steps * sim_substeps      # 36 * 8 steps
    # sim_dt = frame_dt / sim_substeps            # 1 step: 1/60*8 seconds

    sim_fps = 60.0
    sim_substeps = 8 # 업데이트 할 때 64번 서브스텝 진행 후 렌더링, 반복
    sim_duration = 5.0
    sim_frames = int(sim_duration * sim_fps) # 300 frames
    sim_dt = (1.0 / sim_fps) / sim_substeps  # 1 step 지나는데 걸리는 시간
    
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
        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=1.0)
        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = wp.vec3(0.0, 0.0, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, device="cuda", requires_grad=True)

        self.states = []
        for i in range(self.sim_frames * self.sim_substeps + 1):
            self.states.append(self.model.state(requires_grad=True))
        
        wp.sim.collide(self.model, self.states[0])

    
    @wp.kernel
    def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
        delta = pos[0] - target
        loss[0] = wp.dot(delta, delta)


    @wp.kernel
    def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
        tid = wp.tid()

        # gradient descent step
        x[tid] = x[tid] - grad[tid] * alpha


    def render(self, iter):
        for i in range(0, self.sim_frames * self.sim_substeps, self.sim_substeps):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.states[i])
            self.renderer.end_frame()
        
            self.render_time += self.sim_dt * self.sim_substeps


    def train_graph(self):
        ##### start creating graph #####
        wp.capture_begin()

        tape = wp.Tape()

        # forward simulation
        with tape:
            # run control loop
            for i in range(self.sim_frames * self.sim_substeps):
                self.states[i].clear_forces()
                self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

            # compute loss on final state
            wp.launch(
                kernel=self.loss_kernel,
                dim=1, 
                inputs=[self.states[-1].particle_q, self.target, self.loss], 
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

                # # gradient descent step
                # x = self.states[0].particle_qd
                # wp.launch(
                #     kernel=self.step_kernel, 
                #     dim=len(x), 
                #     inputs=[x, x.grad, self.train_rate], 
                #     device="cuda"
                # )
                # x_grad = tape.gradients[self.states[0].particle_qd]
                
                # # debug
                # print(f"Iter: {i} Loss: {self.loss}")
                # print(f"   x: {x} g: {x_grad}")

                # # clear grads for next iteration
                # tape.zero()
            
            with wp.ScopedTimer("Render"):
                self.render(i)


    # def update(self):
    #     with wp.ScopedTimer("simulate", active=False):
    #         wp.sim.collide(self.model, self.state_0)

    #         for s in range(self.sim_substeps):
    #             self.state_0.clear_forces()
    #             self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
    #             (self.state_0, self.state_1) = (self.state_1, self.state_0) # swap states

    # def render(self, is_live=False):
    #     with wp.ScopedTimer("render", detailed=False):
    #         time = 0.0 if is_live else self.sim_time

    #         self.renderer.begin_frame(time)
    #         self.renderer.render(self.state_0)
    #         self.renderer.end_frame()
        
    #     self.sim_time += 1.0 / self.sim_fps



if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/basic.usd")
    example = Example(stage_path)
    example.train_graph()
    example.renderer.save()