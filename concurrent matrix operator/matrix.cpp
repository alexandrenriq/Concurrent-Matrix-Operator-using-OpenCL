//
//  matrix.cpp
//  concurrent matrix operator
//
//  Created by Enrique Alexandre Burga on 6/14/19.
//  Copyright © 2019 Enrique Alexandre Burga. All rights reserved.
//

#include "matrix.hpp"

Matrix::Matrix(int i, int j, float* mtx_input, dispatch_queue_t queue) : di(i), dj(j) {
    this->mtx = (float*)malloc(sizeof(float) * this->di * this->dj);
    this->queue = queue;
    for(int idx=0; idx<i*j; ++idx)
        this->mtx[idx] = mtx_input[idx];
}

Matrix::Matrix(int i, int j, float* mtx_input) : di(i), dj(j) {
    this->mtx = (float*)malloc(sizeof(float) * di * dj);
    for(int idx=0; idx<i*j; ++idx)
        this->mtx[idx] = mtx_input[idx];
}

Matrix::Matrix(Matrix m, dispatch_queue_t queue) {
    this->mtx = m.mtx;
    this->di = m.di;
    this->dj = m.dj;
    this->queue = queue;
}

Matrix::Matrix(int i, int j) : di(i), dj(j) { }

Matrix::Matrix(int i, int j, size_t byte_size) : di(i), dj(j) {
    mtx = (float*)malloc(byte_size);
}

Matrix::~Matrix() {
    mtx = NULL;
}

Matrix Matrix::operator+(Matrix matrix_2) {
    Matrix matrix_out(this->di, this->dj, byte_size);
    
    void *mem_m1 = gcl_malloc(byte_size, this->mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_m2 = gcl_malloc(byte_size, matrix_2.mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_out = gcl_malloc(byte_size, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(this->queue, ^{
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(matrix_sum_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof wgs, &wgs, NULL);
        cl_ndrange range = {
            1,
            {0, 0, 0},
            {MAX_SIZE, 0, 0},
            {wgs, 0, 0}
        };
        matrix_sum_kernel(&range, (cl_float*)mem_m1, (cl_float*)mem_m2, (cl_float*)mem_out);
        gcl_memcpy(matrix_out.mtx, mem_out, byte_size);
    });
    
    gcl_free(mem_m1);
    gcl_free(mem_m2);
    gcl_free(mem_out);
    
    return matrix_out;
}

Matrix Matrix::operator-(Matrix matrix_2) {
    Matrix matrix_out(this->di, this->dj, byte_size);
    
    void *mem_m1 = gcl_malloc(byte_size, this->mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_m2 = gcl_malloc(byte_size, matrix_2.mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_out = gcl_malloc(byte_size, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(this->queue, ^{
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(matrix_substraction_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof wgs, &wgs, NULL);
        cl_ndrange range = {
            1,
            {0, 0, 0},
            {MAX_SIZE, 0, 0},
            {wgs, 0, 0}
        };
        matrix_substraction_kernel(&range, (cl_float*)mem_m1, (cl_float*)mem_m2, (cl_float*)mem_out);
        gcl_memcpy(matrix_out.mtx, mem_out, byte_size);
    });
    
    gcl_free(mem_m1);
    gcl_free(mem_m2);
    gcl_free(mem_out);
    
    return matrix_out;
}

Matrix Matrix::operator%(Matrix matrix_2) {
    Matrix matrix_out(this->di, this->dj, byte_size);
    
    void *mem_m1 = gcl_malloc(byte_size, this->mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_m2 = gcl_malloc(byte_size, matrix_2.mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_out = gcl_malloc(byte_size, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(this->queue, ^{
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(matrix_dot_product_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof wgs, &wgs, NULL);
        cl_ndrange range = {
            1,
            {0, 0, 0},
            {MAX_SIZE, 0, 0},
            {wgs, 0, 0}
        };
        matrix_dot_product_kernel(&range, (cl_float*)mem_m1, (cl_float*)mem_m2, (cl_float*)mem_out);
        gcl_memcpy(matrix_out.mtx, mem_out, byte_size);
    });
    
    gcl_free(mem_m1);
    gcl_free(mem_m2);
    gcl_free(mem_out);
    
    return matrix_out;
}

Matrix Matrix::operator*(float scalar) {
    Matrix matrix_out(this->di, this->dj, byte_size);
    
    void *mem_m1 = gcl_malloc(byte_size, this->mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_m2 = gcl_malloc(sizeof(cl_float), &scalar, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_out = gcl_malloc(byte_size, NULL, CL_MEM_WRITE_ONLY);
    
    dispatch_sync(this->queue, ^{
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(matrix_scalar_multiplication_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof wgs, &wgs, NULL);
        cl_ndrange range = {
            1,
            {0, 0, 0},
            {MAX_SIZE, 0, 0},
            {wgs, 0, 0}
        };
        matrix_scalar_multiplication_kernel(&range, (cl_float*)mem_m1, (cl_float*)mem_m2, (cl_float*)mem_out);
        gcl_memcpy(matrix_out.mtx, mem_out, byte_size);
    });
    
    gcl_free(mem_m1);
    gcl_free(mem_m2);
    gcl_free(mem_out);
    
    return matrix_out;
}

Matrix Matrix::operator*(Matrix matrix_2) {
    Matrix matrix_out(this->di, matrix_2.dj, byte_size);
    
    void *mem_m1 = gcl_malloc(byte_size, this->mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_m2 = gcl_malloc(byte_size, matrix_2.mtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *mem_out = gcl_malloc(byte_size, NULL, CL_MEM_WRITE_ONLY);
    
    size_t vi = matrix_out.di;
    size_t vj = matrix_out.dj;
    dispatch_sync(queue, ^{
        cl_ndrange range = {
            2,
            {0, 0, 0},
            {vi, vj, 0},
            {1, 1, 0}
        };
        matrix_multiplication_kernel(&range, (cl_float*)mem_m1, (cl_float*)mem_m2, (cl_float*)mem_out, (cl_int)this->dj, (cl_int)matrix_2.dj);
        gcl_memcpy(matrix_out.mtx, mem_out, byte_size);
    });
    
    return matrix_out;
}
