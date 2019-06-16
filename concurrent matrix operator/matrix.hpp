//
//  matrix.hpp
//  concurrent matrix operator
//
//  Created by Enrique Alexandre Burga on 6/14/19.
//  Copyright © 2019 Enrique Alexandre Burga. All rights reserved.
//

#ifndef matrix_hpp
#define matrix_hpp

#include <stdio.h>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#include "matrix_sum.cl.h"
#include "matrix_substraction.cl.h"
#include "matrix_dot_product.cl.h"
#include "matrix_scalar_multiplication.cl.h"
#include "matrix_multiplication.cl.h"

#define MAX_SIZE 4096
const size_t byte_size = sizeof(cl_float) * MAX_SIZE;

struct Matrix {
    int di, dj;
    float *mtx;
    dispatch_queue_t queue;
    Matrix(int i, int j, float* mtx_input, dispatch_queue_t queue);
    Matrix(Matrix m, dispatch_queue_t queue);
    Matrix(int i, int j, float* mtx_input);
    Matrix(int i, int j);
    Matrix(int i, int j, size_t byte_size);
    ~Matrix();
    
    // Addition
    Matrix operator+(Matrix matrix_2);
    
    // Substraction
    Matrix operator-(Matrix matrix_2);
    
    // Dot product
    Matrix operator%(Matrix matrix_2);
    
    // Scalar multiplication
    Matrix operator*(float scalar);
    
    // Matrix multiplication
    Matrix operator*(Matrix matrix_2);
};

#endif /* matrix_hpp */
