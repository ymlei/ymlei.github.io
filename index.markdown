---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <img src="assets/profile.png" alt="Profile Photo" class="profile-image" style="width: 250px; height: auto; margin-right: 20px;">
    <div>
        <div style="margin-bottom: 10px;"><a href="https://www.linkedin.com/in/yiming-lei-939658180/">LinkedIn</a></div>
        <div style="margin-bottom: 10px;"><a href="https://scholar.google.com/citations?user=38hLSOgAAAAJ&hl=en">Google Scholar</a></div>
        <div><a href="https://github.com/ymlei">GitHub</a></div>
    </div>
</div>


I work on making datacenter systems fast, efficient, and resilient — from **optical datacenter networks** to **distributed ML training**. I’m a PhD student at the Max Planck Institute for Informatics ([MPI-INF](https://www.mpi-inf.mpg.de/home)), advised by [Yiting Xia](https://sites.google.com/view/yitingxia/home?authuser=0). I work across the stack, including training frameworks, collectives and parallelism, networking stacks, precise time synchronization, and FPGA acceleration.

Highlights
==========
* **Publications:** 2× NSDI first-author papers ([OpenOptics](assets/OpenOptics_CR.pdf) — democratizing optical DCNs; [SyncWise](assets/SyncWise_CR.pdf) — sub-10ns clock accuracy).
* **Open-source systems:** [OpenOptics](https://openoptics.mpi-inf.mpg.de/) — primary contributor, ~30k LoC, modular architecture, comprehensive docs, [tutorial at SIGCOMM’25](https://conferences.sigcomm.org/sigcomm/2025/).
* **ML systems:** Phoenix (checkpoint-less JAX recovery, AWS AI internship) and a feature upstreamed in [JAX PR #36613](https://github.com/jax-ml/jax/pull/36613).

News
======
* [OpenOptics](assets/OpenOptics_CR.pdf) ([website](https://openoptics.mpi-inf.mpg.de/)) has been accepted at **NSDI'26**!
* [SyncWise](assets/SyncWise_CR.pdf) has been accepted at **NSDI'26**!
* We hosted a tutorial on [OpenOptics](https://openoptics.mpi-inf.mpg.de/) at [SIGCOMM'25](https://conferences.sigcomm.org/sigcomm/2025/).

Selected Publications
======================

* **[Under Submission]** Phoenix: Checkpoint-less Failure Recovery for Auto-parallelism.

* **[NSDI’26]** OpenOptics: Enabling Open Research and Implementation of Optical Data Center Networks. ([paper](assets/OpenOptics_CR.pdf), [website](https://openoptics.mpi-inf.mpg.de/))<br>
**Yiming Lei**, Federico De Marchi, Raj Joshi, Jialong Li, Balakrishnan Chandrasekaran, Yiting Xia.

* **[NSDI’26]** SyncWise: Error-Aware Time Synchronization for Reconfigurable Data Center Networks. ([paper](assets/SyncWise_CR.pdf))<br>
**Yiming Lei**, Jialong Li, Zhengqing Liu, Raj Joshi, Yiting Xia.

* **[HotNets’22]** Efficient Flow Scheduling in Distributed Deep Learning Training with Echelon Formation. ([paper](https://dl.acm.org/doi/pdf/10.1145/3563766.3564096))<br>
Rui Pan\*, **Yiming Lei**\*, Jialong Li, Zhiqiang Xie, Binhang Yuan, Yiting Xia. (\*Equal Contributions).

Engineering
=========

<div style="display: flex; align-items: flex-start; margin-bottom: 16px;">
  <img src="assets/logo.png" alt="OpenOptics Logo" style="width: 150px; height: auto; margin-right: 20px;">
  <div>
    <p>
      <strong><a href="https://openoptics.mpi-inf.mpg.de/">OpenOptics</a></strong> (<a href="https://github.com/mpi-ncs/openoptics">GitHub</a>, <a href="assets/OpenOptics_CR.pdf">paper</a>, NSDI'26) — realize customized optical data center networks with ~10 lines of Python.
    </p>
    <ul>
      <li><strong>Primary contributor — shipped ~30k LoC.</strong></li>
      <li>Modular architecture spanning topology, routing, and monitoring, with multi-backend support: Mininet, ns-3, and Tofino.</li>
      <li>Comprehensive documentation; tutorial given at <a href="https://conferences.sigcomm.org/sigcomm/2025/">SIGCOMM'25</a>.</li>
    </ul>
  </div>
</div>

**Phoenix** *(Under Submission)* — Checkpoint-less failure recovery for JAX auto-parallelism. Built during my AWS AI internship; recovers GSPMD/pjit training without periodic checkpointing.

**JAX upstream contribution** — [jax-ml/jax#36613](https://github.com/jax-ml/jax/pull/36613): added a `ProcessFailureError` to JAX’s `live_devices` context manager so callers can identify which devices died on a failure — a primitive that resilient training systems build on top of.

Other Projects
=========

[Digital Molecular Computer](https://www.youtube.com/watch?v=QWBxIEiYPYo&ab_channel=AlexLei) - A specialized processor for boolean satisfiability problem (SAT) inspired by molecular computing. Prototyped with Verilog and FPGA.

Experience
======
* Oct 2021 – Present<br>PhD Student, [Max Planck Institute for Informatics](https://www.mpi-inf.mpg.de/home)
* **Sep 2024 – Mar 2025**<br>**Applied Scientist Intern, AWS AI** — built checkpoint-less failure recovery for JAX-based LLM training (Phoenix).
* Jul 2020 – Mar 2021<br>Research Assistant, [University of Illinois Urbana-Champaign](https://illinois.edu/)
* Sep 2019 – Feb 2020<br>Exchange Student, [Institut supérieur d’électronique de Paris (ISEP)](https://en.isep.fr/)
* Sep 2017 – Jun 2021<br>B.Sc in Computer Science, [Beijing University of Posts and Telecommunications](https://en.wikipedia.org/wiki/Beijing_University_of_Posts_and_Telecommunications)


Misc.
=======
Outside the office, you’ll often find me playing tennis, bouldering, hiking, experimenting in the kitchen, or hanging out with my cat.

<img src="assets/mengmeng.jpeg" alt="Mengmeng" class="profile-image" style="width: 250px; height: auto;">