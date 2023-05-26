#include<stdio.h>

int main(){
    double a[4]={1.0, 2.0 ,3.0 ,4.0};
    char c = 's';
    double b = 23.23;
    double *ptra;
    ptra=&a[0];
    printf("a %f \n",*ptra);
    for(int i=0;i<4;i++){
        ptra=&a[i];
        printf("i %i *ptra %f \n",i,*ptra);
        printf("i %i a[i] %f \n",i,a[i]);
        }
    return 1;
}