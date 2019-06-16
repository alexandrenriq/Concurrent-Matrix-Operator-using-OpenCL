//
//  main.cpp
//  concurrent matrix operator
//
//  Created by Enrique Alexandre Burga on 6/13/19.
//  Copyright © 2019 Enrique Alexandre Burga. All rights reserved.
//

#include <stdio.h>
#include <cstring>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#define CL_SILENCE_DEPRECATION
#include "matrix.hpp"

bool exceed_size(int sz) {
    return sz > MAX_SIZE;
}

static void print_device_info(cl_device_id device) {
    char name[128], vendor[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    // clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    printf("Your machine is working on %s\n", name);
}

void print_matrix(Matrix m) {
    for(int i=0; i<m.di * m.dj; ++i) {
        printf("%d ", (int)m.mtx[i]);
        if((i + 1) % m.dj == 0) printf("\n");
    }
}

Matrix input_matrix(dispatch_queue_t queue) {
    int i, j;
    printf("input dimension i: ");
    scanf("%d", &i);
    printf("input dimension j: ");
    scanf("%d", &j);
    float *m = (float*)malloc(sizeof(float) * i * j);
    m[0] = 1;
    for(int idx=0, k=0; k<i; ++k) {
        for(int l=0; l<j; ++l, ++idx) {
            printf("input value %d-%d: ", k+1, l+1);
            scanf("%f", &m[idx]);
        }
    }
    Matrix out = Matrix(i, j, m, queue);
    return out;
}

int main(int argc, const char * argv[]) {
    
    dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    if(queue == NULL)
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
    print_device_info(gpu);

    int operation;
    while(true) {
        printf("\nPlease add an operation:\n");
        printf("1. Matrix addition\n2. Matrix substraction\n3. Matrix-scalar multiplication\n4. Matrix dot product\n5. Matrix multiplication\n0. Exit\n");
        printf("Operation: ");
        scanf("%d", &operation);
        bool finish = false;
        switch (operation) {
            case 0: {
                finish = true;
                printf("\n\n\nbye\n");
                break;
            }
            case 1: {
                printf("Matrix 1\n");
                auto m1 = input_matrix(queue);
                printf("Matrix 2\n");
                auto m2 = input_matrix(queue);
                
                if(m1.di != m2.di || m1.dj != m2.dj) {
                    printf("\nSorry, dimensions do not match\n");
                    break;
                }
                if(exceed_size(m1.di * m1.dj)) {
                    printf("\nConcurrent memory limit exceed\n");
                    break;
                }
                
                auto sum = m1 + m2;
                printf("\n||Matrix Output||\n");
                print_matrix(sum);
                
                break;
            }
            case 2: {
                printf("Matrix 1\n");
                auto m1 = input_matrix(queue);
                printf("Matrix 2\n");
                auto m2 = input_matrix(queue);
                
                if(m1.di != m2.di || m1.dj != m2.dj) {
                    printf("\nSorry, dimensions do not match\n");
                    break;
                }
                if(exceed_size(m1.di * m1.dj)) {
                    printf("\nConcurrent memory limit exceed\n");
                    break;
                }
                
                auto subs = m1 - m2;
                printf("\n||Matrix Output||\n");
                print_matrix(subs);
                
                break;
            }
            case 3: {
                printf("Matrix 1\n");
                auto m1 = input_matrix(queue);
                int scalar;
                printf("Scalar: ");
                scanf("%d", &scalar);
                
                if(exceed_size(m1.di * m1.dj)) {
                    printf("\nConcurrent memory limit exceed\n");
                    break;
                }
                
                auto result = m1 * scalar;
                printf("\n||Matrix Output||\n");
                print_matrix(result);
                
                break;
            }
            case 4: {
                printf("Matrix 1\n");
                auto m1 = input_matrix(queue);
                printf("Matrix 2\n");
                auto m2 = input_matrix(queue);
                
                if(m1.di != m2.di || m1.dj != m2.dj) {
                    printf("\nSorry, dimensions do not match\n");
                    break;
                }
                if(exceed_size(m1.di * m1.dj)) {
                    printf("\nConcurrent memory limit exceed\n");
                    break;
                }
                
                auto dot_product = m1 % m2;
                printf("\n||Matrix Output||\n");
                print_matrix(dot_product);
                
                break;
            }
            case 5: {
                printf("Matrix 1\n");
                auto m1 = input_matrix(queue);
                printf("Matrix 2\n");
                auto m2 = input_matrix(queue);
                
                if(m1.dj != m2.di) {
                    printf("\nSorry, dimensions do not match\n");
                    break;
                }
                if(exceed_size(m1.di * m1.dj) || exceed_size(m2.di * m2.dj)) {
                    printf("\nConcurrent memory limit exceed\n");
                    break;
                }
                
                auto mult = m1 * m2;
                printf("\n||Matrix Output||\n");
                print_matrix(mult);
                
                break;
            }
            default: {
                printf("\n---Please add a correct operation code---");
                break;
            }
        }
        if(finish) break;
        /*printf("\nPress any key to continue");
        int dummy;
        scanf("%d", dummy);
        printf("\n\n\n");*/
    }

    return 0;
}
