kernel void matrix_multiplication(global float* m_1, global float* m_2, global float* m_out, int cols1, int cols2)
{
    int ix = get_global_id(1);
    int iy = get_global_id(0);
    int idx1 = iy * cols1;
    int idx2 = ix;
    
    float res = 0.0f;
    //res = idx2;

    for(int i=0; i<cols1; ++i) {
        res += m_1[idx1] * m_2[idx2];
        idx1 += 1;
        idx2 += cols2;
    }
    m_out[iy * cols2 + ix] = res;
}
