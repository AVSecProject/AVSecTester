# TODO

This page lists capabilities to implement, grouped by the AVSecTester software component that
owns them. Other documents link to these component sections.

## Datasets

- [x] **NuRec dataset** ported to NuRec and Alpamayo.
- [x] **Carla scenarios** ported with Carla 1.0.
- [ ] **Scenic** for picking test scenarios from full datasets. On NuRec, Scenic should help build the scenario filter and guide object adjusts to create desired scenarios. On Carla, Scenic has been integrated for building Carla scenarios.

## Simulators

- [x] **Port Carla 1.0.**
- [x] **Port NVIDIA Omniverse NuRec (AlpaSim).**
- [x] **Simulation mode: nurec-native.** Applying AlpaSim simualtion on NuRec data. It is suitable for lightweight AI model testing especially end-to-end driving models like Alpamayo.
- [x] **Simulation mode: carla-native.** Applying Carla simualtion on Carla scenario data. It is suitable for complete AV stack testing like Autoware and Apollo, remaining the flexibility to build and control Carla scenarios.
- [x] **Simulation mode: carla-nurec.** Applying Carla's NuRec rendering mode on NuRec data. It is suitable for testing complete AV stack while using the real-world data.
- [ ] **Object insertion to NuRec.** Consider multi-modal data including camera, LiDAR, and radar.
- [ ] **Reactive traffic in NuRec.** Using motion/behavior models to control critical traffic agents.
- [ ] **Realistic time control.** Performance of real systems depends on software/AI component execution time, which is non-trivial to simulate.

## Driving systems

- [x] **Alpamayo** ported to nurec-native.
- [x] **Autoware** ported to carla-native and carla-nurec.
- [ ] **Apollo** ported to carla-native and carla-nurec.

## Attacks

- [x] **Image patch attack.**
- [ ] **LiDAR spoofing attack.**
- [ ] **GPS spoofing attack.**
- [ ] **mmWave radar spoofing attack.**
- [ ] **Physical object attack.**
- [ ] **Agent motion attack.**
- [ ] **Metadata builder.** Building applicable scenarios with associated attack targets, strategies, and parameters. Connected to Scenic interface.
- [ ] **Adaptive attack.** Controlling attack parameters or strategy to maximize attack impact.
- [ ] **Benchmarking.** We should formulate attack categories and generate benchmarks for each (e.g., fixed metadata that can support fair comparison).

## Evaluation

- [ ] **Leaderboard!**
- [ ] **External benchmark integrations.**

## Defenses

- [ ] **Implement typical defenses.** Open the interface for custom implementation.

## Validation

- [ ] **Sim-to-real fidelity.**.
