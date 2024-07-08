# FLT
A Tool for FL

This is a project to make a FL enable on a single(or multiple) server.

Without using physical device setup (which is alot of work).

`FLT` is looking forward to mimic resource limits(vRam, cpu clk, ram, memory, network spec... etc).

Making experiment setup alot easier for FL/CL.

# Notice
- Major update alert : Whole system will be written using `Kubernetes`

# Environment
- python 3.10.0
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

# TODOs
- Development plan changed to developing 'kubernetes' 

# 실험 목록
- 모두가 같은 epoch로 학습한 결과보기
- 특정 device는 더 높은 epoch로 학습한 결과값
  - 더 높은 epoch로 학습한 client에 대해서 가중치 부여하는 상태로 실험해보기
  - 더 높은 epoch로 학습한 client에 대해서 가중치 부여하지 않은 상태로 실험해보기