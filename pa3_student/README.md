## Cloth Simulation using Mass-Spring System

Reference
- https://www.ics.uci.edu/~shz/courses/cs114/docs/proj3/index.html

Resource
- [commit link](https://github.com/hjyu94/cloth_simulation/commit/534370b3ce0d29ced270de8bae6f17bad2815001)

My own implementation
- [commit link](https://github.com/hjyu94/cloth_simulation/commit/65b9464fd0a6ff68cd853a2328654d6154e52231)
- spring internal force, gravity, damping, viscous fluid
  - spring internal force (by Hooke's law)
    - structural spring
    - shear spring
    - flexion spring (bend)
- symplectic time integration
  - v(t+h) = v(t) + h * net_force(t) / m
  - x(t+h) = x(t) + h * v(t+h)

Results
- Parameters
  - mesh resolution: 25*25
  - structural stiffness: 1e3
  - shear stiffness: 1e3
  - bend stiffness: 1e3
  - damping: 0.5
  - viscous: 1.2
  - gravity acceleration: -0.98
- Normal
  - ![1](https://github.com/hjyu94/cloth_simulation/assets/40685291/6cc21e14-3fef-426f-bb6f-6fe7d8d902b9)
  - ![2](https://github.com/hjyu94/cloth_simulation/assets/40685291/1aa09d74-ed41-4497-84d2-6cb63493ab6c)
  - ![3](https://github.com/hjyu94/cloth_simulation/assets/40685291/48f951e9-6a87-4bb0-ad69-142cd3c8f4c1)
- Wire-frame
  - ![2-1](https://github.com/hjyu94/cloth_simulation/assets/40685291/29c9f5bd-a9ba-49b0-a242-88062db9ef20)
  - ![2-2](https://github.com/hjyu94/cloth_simulation/assets/40685291/8ee4ad49-6abe-4249-8593-db1c3aa2b150)
  - ![2-3](https://github.com/hjyu94/cloth_simulation/assets/40685291/76348acc-b2a5-48f7-871c-9592c221d1b9)
