# create_blur_windows_dataset.py

Script para criar uma copia degradada de imagens de um dataset, preservando os nomes dos arquivos e a estrutura esperada pelo pipeline. Ele foi pensado para testar robustez da politica multicamera do SDV-LOAM, simulando cameras degradadas por janelas de frames.

O script nao altera o dataset original. Ele cria uma nova pasta de saida com imagens copiadas ou degradadas e salva um manifesto CSV indicando quais frames foram modificados.

## Requisitos

Execute a partir da pasta de scripts ou informe o caminho completo do script:

```bash
cd /home/analeticiaromero/SDV-LOAM_multi_abril/src/sdv_loam/scripts
```

Dependencias:

```bash
python3 -m pip install pillow numpy
```

`numpy` e opcional, mas recomendado para gerar ruido gaussiano de forma eficiente.

## Modos De Uso

### Uma Pasta De Imagens

Use este modo quando quiser degradar diretamente uma pasta que contem imagens:

```bash
python3 create_blur_windows_dataset.py \
  --source /caminho/para/imagens_originais \
  --output /caminho/para/imagens_degradadas \
  --effect blur_dark \
  --interval 300:520 \
  --overwrite
```

### Dataset Estilo KITTI-360

Use este modo quando o dataset possui pastas de camera, por exemplo `image_00/data_rect` e `image_01/data_rect`:

```bash
python3 create_blur_windows_dataset.py \
  --source-root /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/KITTI-360/data_2d_raw/2013_05_28_drive_0004_sync \
  --output-root /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/DATASETS_DEGRADADOS/degraded_0004 \
  --camera image_00 \
  --effect mixed \
  --interval 300:520 \
  --radius 5 \
  --brightness 0.45 \
  --contrast 0.55 \
  --noise-sigma 22 \
  --occlusion 0.25 \
  --overwrite
```

A entrada esperada nesse exemplo e:

```text
.../2013_05_28_drive_0004_sync/image_00/data_rect/*.png
```

A saida gerada sera:

```text
.../degraded_0004/image_00/data_rect/*.png
```

## Degradar Duas Cameras

Para degradar `image_00` e `image_01` com as mesmas configuracoes:

```bash
python3 create_blur_windows_dataset.py \
  --source-root /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/KITTI-360/data_2d_raw/2013_05_28_drive_0004_sync \
  --output-root /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/DATASETS_DEGRADADOS/degraded_0004_two_cams \
  --camera image_00 \
  --camera image_01 \
  --effect mixed \
  --interval 300:520 \
  --radius 5 \
  --brightness 0.45 \
  --contrast 0.55 \
  --noise-sigma 22 \
  --occlusion 0.25 \
  --overwrite
```

## Janelas Aleatorias

Para gerar janelas aleatorias reproduziveis, use `--random-windows` com `--seed`:

```bash
python3 create_blur_windows_dataset.py \
  --source-root /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/KITTI-360/data_2d_raw/2013_05_28_drive_0004_sync \
  --output-root /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/DATASETS_DEGRADADOS/degraded_0004_random \
  --camera image_00 \
  --camera image_01 \
  --effect mixed \
  --random-windows 4 \
  --random-min-length 40 \
  --random-max-length 80 \
  --radius 5 \
  --brightness 0.45 \
  --contrast 0.55 \
  --noise-sigma 22 \
  --occlusion 0.25 \
  --seed 42 \
  --overwrite
```

Com o mesmo `--seed`, o script gera as mesmas janelas aleatorias.

## Janelas Periodicas

Para degradar 30 frames a cada 200 frames, iniciando no frame 50:

```bash
python3 create_blur_windows_dataset.py \
  --source-root /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/KITTI-360/data_2d_raw/2013_05_28_drive_0004_sync \
  --output-root /media/analeticiaromero/ExtremeSSD/ANA/kitti-360/DATASETS_DEGRADADOS/degraded_0004_periodic \
  --camera image_00 \
  --effect blur_dark \
  --period 200 \
  --length 30 \
  --offset 50 \
  --overwrite
```

## Efeitos Disponiveis

- `blur`: aplica apenas blur gaussiano.
- `blur_dark`: aplica blur, reduz brilho e reduz contraste.
- `noise`: adiciona ruido gaussiano.
- `occlusion`: adiciona retangulos de oclusao.
- `mixed`: combina `blur_dark`, `noise` e `occlusion`.

## Parametros Principais

- `--interval START:END`: degrada uma janela fixa de frames, inclusive `START` e `END`. Pode ser repetido.
- `--random-windows N`: cria `N` janelas aleatorias.
- `--random-min-length`: tamanho minimo das janelas aleatorias.
- `--random-max-length`: tamanho maximo das janelas aleatorias.
- `--period`: periodo para janelas periodicas.
- `--length`: quantidade de frames degradados por periodo.
- `--offset`: frame inicial da primeira janela periodica.
- `--radius`: intensidade do blur gaussiano.
- `--brightness`: multiplicador de brilho usado em `blur_dark` e `mixed`.
- `--contrast`: multiplicador de contraste usado em `blur_dark` e `mixed`.
- `--noise-sigma`: intensidade do ruido gaussiano.
- `--occlusion`: fracao aproximada da imagem coberta por oclusao, entre `0` e `0.95`.
- `--occlusion-rectangles`: numero de retangulos de oclusao por imagem.
- `--occlusion-color R G B`: cor dos retangulos de oclusao.
- `--camera-subdir`: subpasta de imagens dentro de cada camera. O padrao e `data_rect`.
- `--overwrite`: permite sobrescrever a pasta de saida.

## Arquivos Gerados

Para cada camera processada, o script gera:

- Imagens copiadas/degradadas na pasta de saida.
- `manifest_degradation_<camera>.csv`, com uma linha por frame.
- `README_degradation.txt`, com resumo da geracao.

No modo de uma unica pasta, o manifesto se chama:

```text
manifest_degradation.csv
```

No modo KITTI-360, por exemplo para `image_00`, o manifesto fica em:

```text
image_00/data_rect/manifest_degradation_image_00.csv
```

## Como Conferir O Resultado

Para contar quantos frames foram degradados em um manifesto:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

manifest = Path("/caminho/para/manifest_degradation_image_00.csv")
with manifest.open() as f:
    rows = list(csv.DictReader(f))

degraded = [r for r in rows if r["degraded"] == "1"]
print("frames:", len(rows))
print("degraded:", len(degraded))
print("first degraded:", degraded[0]["index"] if degraded else "none")
print("last degraded:", degraded[-1]["index"] if degraded else "none")
PY
```

## Cuidados

- O script usa indice sequencial da imagem ordenada pelo nome do arquivo, nao timestamp.
- `--interval 300:520` inclui os frames `300` e `520`.
- Se `--overwrite` for usado, a pasta de saida existente sera removida antes da nova geracao.
- Para testes justos, mantenha o dataset original separado do dataset degradado.
- Para testar troca de camera causada por degradacao, escolha uma janela em que a camera esteja ativa e estavel antes do inicio da degradacao.
