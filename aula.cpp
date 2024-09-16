#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "include/stb/stb_image.h"
//#include "stb/stb_image_resize.h"
#include "include/stb/stb_image_write.h"

int main(){

//iniciar variavel
int width = 0;
int height = 0;
int channels = 0;

//open image
unsigned char* img = stbi_load("madonna.jpg",&width,&height,&channels,4);

if(img == NULL){
    printf("Erro em carregar imagem");
}

for(int i = 0; i<width * height * 4; i+=4){
    img[i + 1] = 155; 
}

//load image
stbi_write_png("output.jpg",width,height,4,img,4 * width);


stbi_image_free(img);



}