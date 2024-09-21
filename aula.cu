#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb/stb_image.h"
#include "include/stb/stb_image_write.h"

// Kernel CUDA para aplicar o filtro da bandeira LGBT
__global__ void applyLGBTFilterCUDA(unsigned char* img, int width, int height, float opacity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int stripeHeight = height / 6;
    int index = (y * width + x) * 4;

    int R, G, B;

    if (y < stripeHeight) {
        R = 255; G = 0; B = 0;
    } else if (y < 2 * stripeHeight) {
        R = 255; G = 140; B = 0;
    } else if (y < 3 * stripeHeight) {
        R = 255; G = 237; B = 0;
    } else if (y < 4 * stripeHeight) {
        R = 0; G = 128; B = 38;
    } else if (y < 5 * stripeHeight) {
        R = 0; G = 76; B = 255;
    } else {
        R = 115; G = 41; B = 130;
    }

    img[index] = (unsigned char)((1 - opacity) * img[index] + opacity * R);
    img[index + 1] = (unsigned char)((1 - opacity) * img[index + 1] + opacity * G);
    img[index + 2] = (unsigned char)((1 - opacity) * img[index + 2] + opacity * B);
}

// Kernel CUDA para aplicar o filtro da bandeira Trans
__global__ void applyTransFilterCUDA(unsigned char* img, int width, int height, float opacity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int stripeHeight = height / 5;
    int index = (y * width + x) * 4;

    int R, G, B;

    if (y < stripeHeight) {
        R = 91; G = 206; B = 250;
    } else if (y < 2 * stripeHeight) {
        R = 245; G = 169; B = 184;
    } else if (y < 3 * stripeHeight) {
        R = 255; G = 255; B = 255;
    } else if (y < 4 * stripeHeight) {
        R = 91; G = 206; B = 250;
    } else {
        R = 245; G = 169; B = 184;
    }

    img[index] = (unsigned char)((1 - opacity) * img[index] + opacity * R);
    img[index + 1] = (unsigned char)((1 - opacity) * img[index + 1] + opacity * G);
    img[index + 2] = (unsigned char)((1 - opacity) * img[index + 2] + opacity * B);
}

// Kernel CUDA para aplicar o filtro da bandeira Bi
__global__ void applyBiFilterCUDA(unsigned char* img, int width, int height, float opacity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int stripeHeight = height / 3;
    int index = (y * width + x) * 4;

    int R, G, B;

    if (y < stripeHeight) {
        R = 214; G = 2; B = 112;
    } else if (y < 2 * stripeHeight) {
        R = 155; G = 79; B = 150;
    } else {
        R = 0; G = 56; B = 168;
    }

    img[index] = (unsigned char)((1 - opacity) * img[index] + opacity * R);
    img[index + 1] = (unsigned char)((1 - opacity) * img[index + 1] + opacity * G);
    img[index + 2] = (unsigned char)((1 - opacity) * img[index + 2] + opacity * B);
}

// Função CPU para aplicar o filtro da bandeira LGBT
void applyLGBTFilterCPU(unsigned char* img, int width, int height, float opacity) {
    int stripeHeight = height / 6;

    for (int y = 0; y < height; y++) {
        int R, G, B;

        if (y < stripeHeight) {
            R = 255; G = 0; B = 0;
        } else if (y < 2 * stripeHeight) {
            R = 255; G = 140; B = 0;
        } else if (y < 3 * stripeHeight) {
            R = 255; G = 237; B = 0;
        } else if (y < 4 * stripeHeight) {
            R = 0; G = 128; B = 38;
        } else if (y < 5 * stripeHeight) {
            R = 0; G = 76; B = 255;
        } else {
            R = 115; G = 41; B = 130;
        }

        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 4;
            img[index] = (unsigned char)((1 - opacity) * img[index] + opacity * R);
            img[index + 1] = (unsigned char)((1 - opacity) * img[index + 1] + opacity * G);
            img[index + 2] = (unsigned char)((1 - opacity) * img[index + 2] + opacity * B);
        }
    }
}

// Função CPU para aplicar o filtro da bandeira Trans
void applyTransFilterCPU(unsigned char* img, int width, int height, float opacity) {
    int stripeHeight = height / 5;

    for (int y = 0; y < height; y++) {
        int R, G, B;

        if (y < stripeHeight) {
            R = 91; G = 206; B = 250;
        } else if (y < 2 * stripeHeight) {
            R = 245; G = 169; B = 184;
        } else if (y < 3 * stripeHeight) {
            R = 255; G = 255; B = 255;
        } else if (y < 4 * stripeHeight) {
            R = 91; G = 206; B = 250;
        } else {
            R = 245; G = 169; B = 184;
        }

        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 4;
            img[index] = (unsigned char)((1 - opacity) * img[index] + opacity * R);
            img[index + 1] = (unsigned char)((1 - opacity) * img[index + 1] + opacity * G);
            img[index + 2] = (unsigned char)((1 - opacity) * img[index + 2] + opacity * B);
        }
    }
}

// Função CPU para aplicar o filtro da bandeira Bi
void applyBiFilterCPU(unsigned char* img, int width, int height, float opacity) {
    int stripeHeight = height / 3;

    for (int y = 0; y < height; y++) {
        int R, G, B;

        if (y < stripeHeight) {
            R = 214; G = 2; B = 112;
        } else if (y < 2 * stripeHeight) {
            R = 155; G = 79; B = 150;
        } else {
            R = 0; G = 56; B = 168;
        }

        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 4;
            img[index] = (unsigned char)((1 - opacity) * img[index] + opacity * R);
            img[index + 1] = (unsigned char)((1 - opacity) * img[index + 1] + opacity * G);
            img[index + 2] = (unsigned char)((1 - opacity) * img[index + 2] + opacity * B);
        }
    }
}

// Função principal para execução dos filtros em CPU e GPU
int main() {
    // Iniciar variáveis
    int width = 0, height = 0, channels = 0;

    // Carregar a imagem
    unsigned char* img_cpu = stbi_load("madonna.jpg", &width, &height, &channels, 4);
    unsigned char* img_gpu = stbi_load("madonna.jpg", &width, &height, &channels, 4);
    
    if (img_cpu == NULL || img_gpu == NULL) {
        std::cerr << "Erro ao carregar a imagem\n";
        return -1;
    }

    float opacity = 0.5f;  // Definir opacidade (50%)

    // Alocar memória para a imagem na GPU
    unsigned char* d_img;
    size_t img_size = width * height * 4 * sizeof(unsigned char);
    cudaMalloc(&d_img, img_size);
    cudaMemcpy(d_img, img_gpu, img_size, cudaMemcpyHostToDevice);

    // Definir dimensões do grid e dos blocos
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // --------- CPU ---------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    applyLGBTFilterCPU(img_cpu, width, height, opacity);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
    std::cout << "Tempo de processamento do filtro LGBT (CPU): " << duration_cpu.count() << " microsegundos" << std::endl;

    // --------- GPU ---------
    auto start_gpu = std::chrono::high_resolution_clock::now();
    applyLGBTFilterCUDA<<<numBlocks, threadsPerBlock>>>(d_img, width, height, opacity);
    cudaDeviceSynchronize();
    auto stop_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu - start_gpu);
    std::cout << "Tempo de processamento do filtro LGBT (GPU): " << duration_gpu.count() << " microsegundos" << std::endl;

    // Copiar a imagem processada na GPU de volta para a CPU
    cudaMemcpy(img_gpu, d_img, img_size, cudaMemcpyDeviceToHost);

    // Salvar as imagens alteradas
    stbi_write_png("output_cpu_lgbt.png", width, height, 4, img_cpu, width * 4);
    stbi_write_png("output_gpu_lgbt.png", width, height, 4, img_gpu, width * 4);

    // Repetir para os outros filtros (Trans e Bi)
    // --------- CPU (Trans) ---------
    start_cpu = std::chrono::high_resolution_clock::now();
    applyTransFilterCPU(img_cpu, width, height, opacity);
    stop_cpu = std::chrono::high_resolution_clock::now();
    duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
    std::cout << "Tempo de processamento do filtro Trans (CPU): " << duration_cpu.count() << " microsegundos" << std::endl;

    // --------- GPU (Trans) ---------
    start_gpu = std::chrono::high_resolution_clock::now();
    applyTransFilterCUDA<<<numBlocks, threadsPerBlock>>>(d_img, width, height, opacity);
    cudaDeviceSynchronize();
    stop_gpu = std::chrono::high_resolution_clock::now();
    duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu - start_gpu);
    std::cout << "Tempo de processamento do filtro Trans (GPU): " << duration_gpu.count() << " microsegundos" << std::endl;

    cudaMemcpy(img_gpu, d_img, img_size, cudaMemcpyDeviceToHost);

    // Salvar as imagens alteradas
    stbi_write_png("output_cpu_trans.png", width, height, 4, img_cpu, width * 4);
    stbi_write_png("output_gpu_trans.png", width, height, 4, img_gpu, width * 4);

    // --------- CPU (Bi) ---------
    start_cpu = std::chrono::high_resolution_clock::now();
    applyBiFilterCPU(img_cpu, width, height, opacity);
    stop_cpu = std::chrono::high_resolution_clock::now();
    duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
    std::cout << "Tempo de processamento do filtro Bi (CPU): " << duration_cpu.count() << " microsegundos" << std::endl;

    // --------- GPU (Bi) ---------
    start_gpu = std::chrono::high_resolution_clock::now();
    applyBiFilterCUDA<<<numBlocks, threadsPerBlock>>>(d_img, width, height, opacity);
    cudaDeviceSynchronize();
    stop_gpu = std::chrono::high_resolution_clock::now();
    duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_gpu - start_gpu);
    std::cout << "Tempo de processamento do filtro Bi (GPU): " << duration_gpu.count() << " microsegundos" << std::endl;

    cudaMemcpy(img_gpu, d_img, img_size, cudaMemcpyDeviceToHost);

    // Salvar as imagens alteradas
    stbi_write_png("output_cpu_bi.png", width, height, 4, img_cpu, width * 4);
    stbi_write_png("output_gpu_bi.png", width, height, 4, img_gpu, width * 4);

    // Limpar memória
    cudaFree(d_img);
    stbi_image_free(img_cpu);
    stbi_image_free(img_gpu);

    return 0;
}