import os
import math

import numpy as np

import torch

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    # f: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    inv_mass: float,
):
    tid = wp.tid()
    # print(x.numpy()[0])

    # v_new = v[tid] + f[tid] * inv_mass * dt + gravity * dt
    v_new = v[tid] + gravity * dt
    x_new = x[tid] + v_new * dt

    v[tid] = v_new
    x[tid] = x_new



class Example:
    def __init__(self, stage, renderer_type="sim"):
        self.sim_fps = 60.0
        self.sim_substeps = 64 # 업데이트 할 때 64번 서브스텝 진행 후 렌더링, 반복
        self.sim_duration = 5.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps  # 초당 60Hz

        self.sim_time = 0.0 # 현재 시뮬레이션 시간, 렌더링 할 때 파라미터로 쓰임
        
        self.renderer_type = renderer_type

        self.create_scene(stage)


    def create_scene(self, stage):
        builder = wp.sim.ModelBuilder()

        # ground
        # builder.add_shape_plane()
        # builder.set_ground_plane()
        
        # cloth
        self.sim_width = 10
        self.sim_height = 3

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

        if self.renderer_type == "sim":
            self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=40.0)

            self.state_0 = self.model.state()
            self.state_1 = self.model.state()

        elif self.renderer_type == "usd":
            self.renderer = wp.render.UsdRenderer(stage)

            self.point_radius = 0.1

        self.integrator = wp.sim.SemiImplicitIntegrator()


    def update(self):
        if self.renderer_type == "sim":
            with wp.ScopedTimer("simulate", active=False):
                wp.sim.collide(self.model, self.state_0)

                for s in range(self.sim_substeps):
                    self.state_0.clear_forces()
                    self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                    (self.state_0, self.state_1) = (self.state_1, self.state_0) # swap states


        elif self.renderer_type == "usd":
            with wp.ScopedTimer("solve", active=False):
                for s in range(self.sim_substeps):
                    wp.launch(
                        kernel=integrate,
                        dim=len(self.model.particle_q),
                        inputs=[self.model.particle_q, self.model.particle_qd, (0.0, -9.8, 0.0), self.sim_dt, 1],
                    )
                

    def render(self, is_live=False):
        if self.renderer_type == "sim":
            with wp.ScopedTimer("render", detailed=False):
                time = 0.0 if is_live else self.sim_time

                self.renderer.begin_frame(time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()
            
            self.sim_time += 1.0 / self.sim_fps
        

        elif self.renderer_type == "usd":
            with wp.ScopedTimer("render", detailed=False):
                time = 0.0 if is_live else self.sim_time
               
                self.renderer.begin_frame(time)
                self.renderer.render_points(
                    points=self.model.particle_q.numpy(), 
                    radius=self.point_radius, 
                    name="points"
                )
                self.renderer.end_frame()

            self.sim_time += 1.0 / self.sim_fps


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/basic.usd")
    example = Example(stage_path, renderer_type="usd")

    for i in range(example.sim_frames):
        example.update()
        example.render()

    example.renderer.save()