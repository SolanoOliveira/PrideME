#include <iostream>
#include <thread>

void call_from_thread(){
    std::cout << "oii" << std::endl;
}


void call_from_thread2(){
    std::cout << "oii2" << std::endl;
}

void call_from_thread3(){
    std::cout << "oii3" << std::endl;
}
void call_from_thread4(){
    std::cout << "oii4" << std::endl;
}

void call_from_thread5(){
    std::cout << "oii5" << std::endl;
}

int soma(){
    return 1+1;
}

int main(){

    std::cout << "oii" << std::endl;

    std::thread t1(call_from_thread2);
    std::thread t2(call_from_thread3);
    std::thread t3(call_from_thread4);
    std::thread t4(call_from_thread5);



    t1.join();
    t2.join();
    t3.join();
    t4.join();


    return 0;
}
