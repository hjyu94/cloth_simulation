import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    # seconds
    sim_duration = 5.0

    # control frequency
    frame_dt = 1.0 / 60.0
    frame_steps = int(sim_duration / frame_dt)

    # sim frequency
    sim_substeps = 64
    sim_steps = frame_steps * sim_substeps
    sim_dt = frame_dt / sim_substeps # 1 step 을 지나는데 걸리는 시간

    render_time = 0.0

    train_iters = 20
    train_rate = 0.001

    ke = 1.0e3
    kf = 0.0
    kd = 1.0e1
    mu = 0.25


    def __init__(self, stage, adapter=None):
        self.device = wp.get_device(adapter)
        self.create_scene(stage)


    def create_scene(self, stage):
        builder = wp.sim.ModelBuilder()

        sim_width = 32
        sim_height = 32
        
        cell_x = 0.5
        cell_y = 0.5
        
        builder.add_cloth_grid(
            pos=(-sim_width * cell_x * 0.5, 20.0, -sim_height * cell_y * 0.5),
            rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.5),
            vel=(0.0, 0.0, 0.0),
            dim_x=sim_width,
            dim_y=sim_height,
            cell_x=cell_x,
            cell_y=cell_y,
            mass=0.1,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e1,
            # tri_lift=10.0,
            # tri_drag=5.0,
        )
        
        # builder.add_particle(
        #     pos=(-5.0, 0.0, 0.0),
        #     vel=(5.0, 20.0, 0.0),
        #     mass=0.1,
        #     radius=1.0
        # )
        # builder.add_shape_sphere(
        #     pos=(0.0, 5.0, 0.0), 
        #     radius=1.0, 
        #     density=10.0, 
        #     body=-1,
        #     ke=self.ke, 
        #     kf=self.kf, 
        #     kd=self.kd, 
        #     mu=self.mu
        # )
        
        self.model = builder.finalize(device=self.device, requires_grad=True)
        self.model.ground = True

        self.model.soft_contact_ke = self.ke
        self.model.soft_contact_kf = self.kf
        self.model.soft_contact_kd = self.kd
        self.model.soft_contact_mu = self.mu
        # self.model.soft_contact_mu.requires_grad = True
        
        self.model.requires_grad = True
        self.model.soft_contact_margin = 10.0
        self.model.soft_contact_restitution = 1.0

        # self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=1.0)
        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=10.0)
        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.target = wp.vec3(10.0, 10.0, 10.0)
        self.com = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        
        self.init_force = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=True)
        self.pick_force = wp.array([5.0, 10.0, 5.0], dtype=wp.vec3, device=self.device, requires_grad=True)
        
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


    @wp.kernel
    def apply_forces(
        particle_f: wp.array(dtype=wp.vec3),
        idx: int,
        pick_force: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        # if tid > 300 and tid < 700:
        
        if idx < 5120 and tid / 33 > 10 and tid / 33 < 22 and tid % 33 > 10 and tid % 33 < 22:
            f = pick_force[0]
        else:
            f = wp.vec3(0.0, 0.0, 0.0)

        # f = pick_force[0]
        particle_f[tid] = f
        
        
    def train_graph(self):
        
        # force = np.zeros((self.states[0].particle_count, 3))
        # for idx in range(300, 700):
        #     if idx % 33 < 10:
        #         force[idx] = (5, 20, 3)
        # self.states[0].particle_f = wp.from_numpy(force, dtype=wp.vec3, device=self.device)

        ##### start creating graph #####
        wp.capture_begin()

        tape = wp.Tape()

        # forward simulation
        with tape:
            # run control loop
            for i in range(self.sim_steps):
                self.states[i].clear_forces()
                wp.launch(
                    kernel=self.apply_forces,
                    dim=self.states[i].particle_count,
                    inputs=[
                        self.states[i].particle_f,
                        i,
                        self.pick_force,
                    ],
                )
                self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

            # compute loss on final state
            self.com.zero_()
            wp.launch(
                self.com_kernel,
                dim=self.model.particle_count,
                inputs=[self.states[-1].particle_q, self.model.particle_count, self.com],
                device=self.device,
            )

            wp.launch(
                kernel=self.loss_kernel,
                dim=1, 
                inputs=[self.com, self.target, self.loss], 
                device=self.device
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

                print(f"======================== Iter: {i} ========================")
                print(f"com: {self.com.numpy()}")
                print(f"pick_force: {self.pick_force.numpy()}")
                print(f"target: {self.target}")
                
                # gradient descent step
                wp.launch(
                    kernel=self.step_kernel, 
                    dim=len(self.pick_force), 
                    inputs=[self.pick_force, self.pick_force.grad, self.train_rate], 
                    device=self.device
                )
                
                # debug
                print(f"loss: {self.loss}")
                print(f"gradient of pick_force: {self.pick_force.grad.numpy()}")
                print(f"pick_force(updated): {self.pick_force.numpy()}")

                # clear grads for next iteration
                tape.zero()
            
            with wp.ScopedTimer("Render"):
                self.render(i)
        
        # wp.capture_launch(self.graph)
        # self.render(0)
        

if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/basic.usd")
    example = Example(stage_path)
    example.train_graph()
    example.renderer.save()