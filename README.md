![output_cpu_lgbt](https://github.com/user-attachments/assets/813bcc79-cbe2-4e79-af21-197dea87ddbc)

# PrideME

Este projeto aplica filtros baseados nas bandeiras **LGBT**, utilizando processamento na **CPU** e **GPU** com **CUDA**. O código carrega uma imagem, aplica os filtros e exibe o tempo de processamento para cada filtro.

## Requisitos

Para compilar e executar este programa, você precisará dos seguintes pacotes e ferramentas:

- **CUDA Toolkit** (para suporte à GPU)
- **g++** ou **nvcc** (compilador)
- **STB Image Library** (inclusa no projeto)

## Instruções de Instalação

### 1. Instalar o CUDA Toolkit

No **Ubuntu**, use:

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

#### 2. Instalar o g++

No Ubuntu, instale o g++ com:

```bash
sudo apt-get install g++
```

##### Compilação

Navegue até o diretório do projeto:

nvcc -o image_filter aula.cu -lstdc++

voce pode trocar a imagem no proprio codigo, lembre-se de por na pasta antes
