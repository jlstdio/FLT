<img src="https://github.com/jlstdio/FLT/assets/35446381/932a2281-0e7d-4210-af00-2694b05f88ac" width="200">

# FLT
A Tool for FL

This is a project to make a FL enable on a single(or multiple) server.

Without using physical device setup (which is alot of work).

`FLT` is looking forward to mimic resource limits(vRam, cpu clk, ram, memory, network spec... etc).

Making experiment setup alot easier for FL/CL.

# Notice
- Major update alert : Whole system will be written using `Kubernetes`

# Environment
- Python 3.9.16
- venv : required modules in requirements.txt
  - `pip install -r requirements.txt`

# Whats available
- Clients & server configuration on a single file : `config.json`
  - Only defined dataset & model, configuration is only file you need to change
- Server
- Limited clients instance
  - Depend on model size (e.g. gtx 2080ti x 4 : 20 clients on `testModel.py` model)
- Gpu sharing for clients
- Federated Learning Optimizer
  - FedAvg
  - TBA
  
# Reference
- [BCE loss VS CE loss](https://kim95175.tistory.com/26)
- [Learning Rate Scheduler](https://wikidocs.net/157282)