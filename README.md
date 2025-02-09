### Data 

1. 서울과학기술대학교 일반 대학원 자율주행 과제 수행 데이터 
[Google Drive](https://drive.google.com/file/d/1U2GsexF012DQ8k0Qee14fKBxwtdIBnzn/view?usp=sharing)
2. KITTI Dataset 
Download KITTI Dataset in homepages down below
[KITTI](https://www.cvlibs.net/datasets/kitti/)

### run
```bash 
python vis.py 
```

### Demo 
![demo](asset/demo.gif)

### run Docker 

```bash 
cd path/to/here 
docker build -t docker-name .

docker run -it --rm  -v /path/to/KiTTi:/workspace/kitti -v /tmp/.X11-unix:/tmp/.X11-unix --volume="$HOME/.Xauthority:/root/.Xauthority:rw"   -e DISPLAY=$DISPLAY   -e XDG_RUNTIME_DIR=/tmp --net=host docker-name 
```

### TODO

- [x] **Implement KITTI in `dataset.py`**  
  - `dataset.py`에서 KITTI 데이터셋 관련 기능 구현 완료.

- [x] **Write Dockerfile: Xforwarding, connect KITTI dataset dir**  
  - **X11 포워딩 설정:**  
    Docker 컨테이너에서 GUI 애플리케이션 실행을 위해 `xhost` 및 `DISPLAY` 변수를 설정해야 함.
  - **KITTI 데이터셋 디렉토리 연결:**  
    호스트 머신의 KITTI 데이터셋 디렉토리를 컨테이너와 공유 (예: `/path/to/kitti:/workspace/kitti`).


- [ ] **Open3d headless 설정 구현:**
  - docker container 상에서 open3d를 돌리려 하니 너무 느리다.