# SDV-LOAM Multi-camera

Adaptacao do modulo visual do [SDV-LOAM](https://github.com/ZikangYuan/SDV-LOAM) para receber varias cameras e um LiDAR. Cada camera mantem seu estado visual; uma camera ativa por vez atualiza a trajetoria do conjunto. Este repositorio nao implementa fusao simultanea das imagens.

## Requisitos e compilação

Versoes presentes no ambiente usado neste projeto (referencia para reproducao, nao exigencia de versao exata):

| Componente | Versao |
| --- | --- |
| Ubuntu | 18.04.6 LTS |
| ROS 1 | Melodic (`roscpp` 1.14.13; `catkin` 0.7.29) |
| GCC / C++ | 9.4.0 / C++14 |
| CMake | 3.16.7 |
| Eigen3 | 3.3.4 |
| PCL | 1.8.1 |
| OpenCV | 3.2.0 |
| Pangolin | 0.6 |
| Boost | 1.65.1 |
| SuiteSparse | 5.1.2 |

O projeto declara CMake >= 2.8.3, C++14 e Pangolin >= 0.2 no `CMakeLists.txt`; as demais versoes acima foram verificadas neste ambiente. Pangolin e OpenCV precisam estar disponiveis para que o executavel `sdv_loam` seja compilado. Execute na raiz do workspace `SDV-LOAM_multi_abril`:

```bash
source /opt/ros/melodic/setup.bash
catkin_make -j4
source devel/setup.bash
```

Os dados e as bags nao acompanham o codigo. Antes de executar, confira os topicos e os caminhos de calibracao no launch escolhido.

## Executar

Em um terminal, inicie o sistema com o launch apropriado:

| Dataset | Launch | Topicos esperados |
| --- | --- | --- |
| KITTI odometry 09 | `run_kitti_09_multicam.launch` | `/image_left`, `/image_right`, `/kitti/velo/pointcloud` |
| KITTI-360 | `run_kitti_360_multicam.launch` | `/image_left`, `/image_right`, `/kitti/velo/pointcloud` |
| nuScenes | `run_nuscenes_6cam.launch` | `/cam_*/raw`, `/lidar_top` |

Por exemplo, para KITTI 09:

```bash
roslaunch sdv_loam run_kitti_09_multicam.launch resultPath:=/caminho/para/output/pred.txt
```

Em outro terminal, publique a bag correspondente:

```bash
source devel/setup.bash
rosbag play /caminho/para/09.bag --clock -d 1.0
```

Crie o diretorio de saida antes de iniciar o launch. O `trackingCameraId=-1` permite selecionar a camera ativa; um ID `0`, `1`, etc. fixa o tracking naquela camera. O launch de nuScenes vem configurado com `trackingCameraId=5`, portanto usa apenas essa camera no tracking, embora liste seis entradas.

## Saídas

Para `resultPath:=/caminho/para/output/pred.txt`, o sistema grava:

| Arquivo | Conteudo |
| --- | --- |
| `pred.txt` | Poses no formato KITTI |
| `pred.tum` | Poses com timestamp no formato TUM |
| `pred_timestamps.txt` | Timestamps das poses |
| `pred_active_camera.txt` | Camera ativa por pose |
| `pred_camera_scores.csv` | Score e diagnosticos por camera e timestamp |
| `pred_timing.txt` | Tempo de processamento |

Para comparar com ground truth no `evo`, use arquivos no mesmo formato e verifique a correspondencia de timestamps ou frames antes de calcular APE/RPE.

## Experimentos

- [`scripts/run_kitti_bag_repeated.sh`](scripts/run_kitti_bag_repeated.sh), [`scripts/run_kitti360_bag_repeated.sh`](scripts/run_kitti360_bag_repeated.sh) e [`scripts/run_nuscenes_bag_repeated.sh`](scripts/run_nuscenes_bag_repeated.sh): repeticoes com bags; consulte `--help` para definir bag, numero de execucoes e pasta de saida.
- [`scripts/run_kitti09_param_sweep.sh`](scripts/run_kitti09_param_sweep.sh): analise de parametros no KITTI 09 com o launch parametrizavel [`run_kitti_09_multicam_param.launch`](launch/run_kitti_09_multicam_param.launch). Por padrao, executa 15 repeticoes por configuracao e nao inclui a configuracao base.
- [`scripts/generate_kitti09_param_metrics.sh`](scripts/generate_kitti09_param_metrics.sh): extrai APE/RPE com `evo` dos resultados dessa analise; requer `evo` instalado e um ground truth KITTI compativel.
- [`scripts/create_blur_windows_dataset.py`](scripts/create_blur_windows_dataset.py): gera copias de imagens com degradacoes; [instrucoes e exemplos](scripts/README_create_blur_windows_dataset.md).

## Referência

Z. Yuan et al., ["SDV-LOAM: Semi-Direct Visual-LiDAR Odometry and Mapping"](https://ieeexplore.ieee.org/abstract/document/10086694), IEEE TPAMI, 2023. Detalhes da adaptacao estao em [`docs/REFATORACAO_MULTI_CAMERA.md`](docs/REFATORACAO_MULTI_CAMERA.md).
