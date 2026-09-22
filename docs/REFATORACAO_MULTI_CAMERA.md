# Refatoracao Multi-Camera

Este documento registra o baseline de refatoracao para a migracao incremental do SDV-LOAM para uma arquitetura multi-camera com trajetoria global unica no frame do LiDAR.

## Objetivo Inicial

As duas primeiras etapas de execucao devem preservar integralmente o comportamento monocular:

1. congelar um baseline monocular reproduzivel;
2. introduzir tipos e estruturas novas sem alterar o fluxo numerico atual.

## Baseline Monocular

Antes de ligar qualquer funcionalidade multi-camera, validar e registrar:

- sequencia usada no teste;
- arquivo de calibracao e arquivo de parametros do sensor;
- arquivo de saida de trajetoria gerado por `printResult`;
- logs principais de tracking e quantidade de keyframes;
- observacao qualitativa de estabilidade do tracking.

## Semantica Atual de Pose

No codigo atual, a pose visual nativa e a pose da camera no mundo:

- `FrameShell::camToWorld`
- `FrameHessian::worldToCam_evalPT`
- `FrameHessian::PRE_camToWorld`
- `FrameHessian::PRE_worldToCam`

A refatoracao vai introduzir `RigState::T_WL` sem substituir imediatamente essa representacao interna.

## Nova Camada de Estado Global

O novo tipo `RigState` deve ser entendido como dono da trajetoria global:

- `T_WL`: pose do LiDAR no mundo;
- `T_LC_i`: extrinseca fixa do LiDAR para cada camera;
- `activeCameraId`: camera que fornece a atualizacao visual completa no instante atual.

Enquanto o monocular legado estiver ativo, o sistema usara apenas a camera `0`.

## Regra de Seguranca de Refatoracao

Nesta fase, nao alterar:

- residuals host-target;
- bundle adjustment visual local;
- politica de marginalizacao;
- matching direto ou propagacao entre frames da mesma camera.

Qualquer mudanca estrutural deve primeiro manter `VisualState[0]` equivalente ao monocular original.
