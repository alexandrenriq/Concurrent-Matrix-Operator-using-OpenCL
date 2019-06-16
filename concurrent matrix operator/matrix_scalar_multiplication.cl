kernel void matrix_scalar_multiplication(global float* m_1, global float* scalar, global float* m_out)
{
    size_t i = get_global_id(0);
    m_out[i] = m_1[i] * (*scalar);
}
