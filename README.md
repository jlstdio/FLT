# FLT
A Tool for FL

# Environment
`pip install torchvision`

`pip install torch`

# TODOs
- Virtual env config
  - [ ] GPU parallel setup
- Edge device config
  - [ ] virtual device spec config
  - [ ] Individual learning phase config
- Protocol config
  - [ ] Parameter uplink to server
  - [ ] Model downlink to edge


# 실험 목록
- 모두가 같은 epoch로 학습한 결과보기
- 특정 device는 더 높은 epoch로 학습한 결과값
  - 더 높은 epoch로 학습한 client에 대해서 가중치 부여하는 상태로 실험해보기
  - 더 높은 epoch로 학습한 client에 대해서 가중치 부여하지 않은 상태로 실험해보기