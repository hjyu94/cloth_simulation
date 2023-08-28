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
        
    frame_dt = 1.0 / 60.0                       # 1 frame: 1/60 seconds
    frame_steps = int(sim_duration / frame_dt)  # 36 frames

    sim_substeps = 8
    sim_steps = frame_steps * sim_substeps      # 36 * 8 steps
    sim_dt = frame_dt / sim_substeps            # 1 step: 1/60*8 seconds
    sim_time = 0.0

    render_time = 0.0

    train_iters = 250
    train_rate = 0.01

    # ke = 1.0e4
    # kf = 0.0
    # kd = 1.0e1
    # mu = 0.25


    def __init__(self, stage):
        self.create_scene(stage)


    def create_scene(self, stage):
        builder = wp.sim.ModelBuilder()

        self.sim_width = 64
        self.sim_height = 32

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

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=1.0)
        
        self.target = wp.vec3(0.0, 0.0, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, device="cuda", requires_grad=True)

        self.states = []
        for i in range(self.sim_steps + 1):
            self.states.append(self.model.state(requires_grad=True))
        
        wp.sim.collide(self.model, self.states[0])



    def update(self):
        with wp.ScopedTimer("simulate", active=False):
            wp.sim.collide(self.model, self.state_0)

            for s in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                (self.state_0, self.state_1) = (self.state_1, self.state_0) # swap states


    def render(self, is_live=False):
        with wp.ScopedTimer("render", detailed=False):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()
        
        self.sim_time += 1.0 / self.sim_fps



if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/basic.usd")
    example = Example(stage_path)

    for i in range(example.sim_frames):
        example.update()
        example.render()

    example.renderer.save()