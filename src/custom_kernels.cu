#include <stdio.h> 
#include <math.h>

__global__ void matmul(float * a, float * b, float * c, int * a_shape, int * b_shape) {
    if ((blockDim.y * blockIdx.y + threadIdx.y) < a_shape[0] && (blockDim.x * blockIdx.x + threadIdx.x) < b_shape[1]) {
        int aMin = (blockDim.y * blockIdx.y + threadIdx.y) * a_shape[1];
        int aMax = (blockDim.y * blockIdx.y + threadIdx.y + 1) * a_shape[1];
        int aStep = 1;
        int bMin = blockDim.x * blockIdx.x + threadIdx.x;
        int bMax = blockDim.x * blockIdx.x + threadIdx.x + b_shape[0] * b_shape[1];
        int bStep = b_shape[1];
        float temp = 0;
        for (int ai = aMin, bi = bMin; ai < aMax && bi < bMax; ai += aStep, bi += bStep) {
            temp += a[ai] * b[bi];
        }
        int a_index = (blockDim.y * blockIdx.y + threadIdx.y) * b_shape[1];
        c[a_index + bMin] = temp;
    }
}

__global__ void transpose(float * a, float * a_T, int * a_shape) {
    int elem_idx = (blockDim.y * blockIdx.y + threadIdx.y) * a_shape[1] + blockDim.x * blockIdx.x + threadIdx.x;
    if (elem_idx < a_shape[0] * a_shape[1]) {
        int a_t_1 = a_shape[0];
        int elem_tr_idx = (blockDim.x * blockIdx.x + threadIdx.x) * a_t_1 + blockDim.y * blockIdx.y + threadIdx.y;
        a_T[elem_tr_idx] = a[elem_idx];
    }

}

__global__ void row_mean(float * a, float * mean, int * a_shape) {
    //Returns a column
    int row_num = (blockDim.x * blockIdx.x + threadIdx.x);
    if (row_num < a_shape[0]) {
        int start_idx = row_num * a_shape[1];
        int end_idx = start_idx + a_shape[1];
        float sum = 0;
        for (int i = start_idx; i < end_idx; i++) {
            sum += a[i];
        }
        mean[row_num] = sum / a_shape[1];
    }
}

__global__ void column_mean(float * a, float * mean, int * a_shape) {
    //Returns a row
    int col_num = (blockDim.x * blockIdx.x + threadIdx.x);
    if (col_num < a_shape[1]) {
        int start_idx = col_num;
        int end_idx = start_idx + a_shape[1] * a_shape[0];
        float sum = 0;
        for (int i = start_idx; i < end_idx; i += a_shape[1]) {
            sum += a[i];
        }
        mean[col_num] = sum / a_shape[0];
    }
}

__global__ void min_row(float * a, int * a_shape, float * min_row, int * arg_min) {
    //Returns a column for min_row and argmin 
    int row_num = (blockDim.x * blockIdx.x + threadIdx.x);
    if (row_num < a_shape[0]) {
        int start_idx = row_num * a_shape[1];
        int end_idx = start_idx + a_shape[1];
        min_row[row_num] = a[start_idx];
        arg_min[row_num] = 0;
        for (int col = start_idx + 1, index = 1; col < end_idx, index < a_shape[1]; col++, index++) {
            if (a[col] < min_row[row_num]) {
                min_row[row_num] = a[col];
                arg_min[row_num] = index;
            }
        }
    }

}

__global__ void sum_axis3(float * a, int * a_shape, float * result) {
    //a[i][j][k] = k+a_shape[2]*j + a_shape[2]*a_shape[1]*i

    int col_num = (blockDim.x * blockIdx.x + threadIdx.x);
    int row_num = (blockDim.y * blockIdx.y + threadIdx.y);
    if (row_num < a_shape[0] && col_num < a_shape[1]) {
        int start_idx = (row_num * a_shape[1] + col_num) * a_shape[2];
        int end_idx = start_idx + a_shape[2];
        int step = 1;
        float temp = 0;
        for (int idx = start_idx; idx < end_idx; idx += step) {
            temp += a[idx];
        }
        result[row_num * a_shape[1] + col_num] = temp;
    }

}

__global__ void sum_axis2(float * a, int * a_shape, float * result) {
    //a[i][j][k] = k+a_shape[2]*j + a_shape[2]*a_shape[1]*i

    int col_num = (blockDim.x * blockIdx.x + threadIdx.x);
    int row_num = (blockDim.y * blockIdx.y + threadIdx.y);
    if (row_num < a_shape[0] && col_num < a_shape[2]) {
        int start_idx = row_num * a_shape[1] * a_shape[2] + col_num;
        int end_idx = start_idx + a_shape[2] * a_shape[1];
        int step = a_shape[2];
        float temp = 0;
        for (int idx = start_idx; idx < end_idx; idx += step) {
            temp += a[idx];
        }
        result[row_num * a_shape[2] + col_num] = temp;
    }

}

__global__ void sum_axis1(float * a, int * a_shape, float * result) {
    //a[i][j][k] = k+a_shape[2]*j + a_shape[2]*a_shape[1]*i

    int col_num = (blockDim.x * blockIdx.x + threadIdx.x);
    int row_num = (blockDim.y * blockIdx.y + threadIdx.y);
    if (row_num < a_shape[1] && col_num < a_shape[2]) {
        int start_idx = (row_num) * a_shape[2] + col_num;
        int end_idx = start_idx + a_shape[2] * a_shape[1] * a_shape[0];
        int step = a_shape[2] * a_shape[1];
        float temp = 0;
        for (int idx = start_idx; idx < end_idx; idx += step) {
            temp += a[idx];
        }
        result[row_num * a_shape[2] + col_num] = temp;
    }

}

__global__ void argmin_mu_diff(float * data, float * mu, int * data_shape, int * mu_shape, int * arg_min) {

    int data_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (data_id < data_shape[0]) {
        int startIdx = (blockDim.x * blockIdx.x + threadIdx.x) * data_shape[1];
        float min_diff = INT_MAX;
        float arg_min_diff = -1;
        for (int i = 0; i < mu_shape[0]; i++) {
            float diff = 0;
            for (int dim = 0; dim < mu_shape[1]; dim++) {
                diff += (data[startIdx + dim] - mu[i * mu_shape[1] + dim]) * (data[startIdx + dim] - mu[i * mu_shape[1] + dim]);
            }
            if (diff < min_diff) {
                min_diff = diff;
                arg_min_diff = i;
            }
        }
        arg_min[data_id] = arg_min_diff;
    }

}