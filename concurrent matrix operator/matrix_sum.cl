kernel void matrix_sum(global float* m_1, global float* m_2, global float* m_out)
{
    size_t i = get_global_id(0);
    m_out[i] = m_1[i] + m_2[i];
}
