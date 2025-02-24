from math import sqrt
from triton import cdiv, jit
import triton.language as tl

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_channel(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, head_dim, clen = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, :, None].expand(
        batch, num_key_value_heads, head_dim, clen, n_rep
    )
    return hidden_states.reshape(batch, num_key_value_heads, head_dim, clen * n_rep)


def repeat_time(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, head_dim, tlen = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, num_key_value_heads, head_dim, n_rep, tlen
    )
    return hidden_states.reshape(batch, num_key_value_heads, head_dim, n_rep * tlen)


@jit
def _fwd_kernel(
    Q,
    Q_bias,
    K_glob,
    Bias_time,
    Bias_chan,
    V,
    time_dropout,
    chan_dropout,
    sm_scale,
    p_drop,
    L,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_q_bias_h,
    stride_q_bias_n,
    stride_k_glob_z,
    stride_k_glob_h,
    stride_k_glob_n,
    stride_k_glob_k,
    stride_bt_z,
    stride_bt_h,
    stride_bt_n,
    stride_bt_k,
    stride_bc_z,
    stride_bc_h,
    stride_bc_n,
    stride_bc_k,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    N_CTX: tl.constexpr,
    Z_HQ_N_CTX: tl.constexpr,
    Z_HKV_N_CTX: tl.constexpr,
    REL_BIAS_TIME: tl.constexpr,
    REL_BIAS_CHAN: tl.constexpr,
    TIME_CTX: tl.constexpr,
    CHANNEL_CTX: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_WINDOWED: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_qz = off_hz // H_Q
    off_qh = off_hz % H_Q
    off_kh = (off_hz // HEAD_RATIO) % H_KV
    time_end_block = tl.cdiv((start_m + 1) * BLOCK_M, CHANNEL_CTX)
    time_beg_block = ((start_m) * BLOCK_M) // CHANNEL_CTX
    last_time_end_block = tl.cdiv(time_end_block * CHANNEL_CTX, BLOCK_M)
    IS_LAST_QUERY = (start_m + 1) * BLOCK_M >= N_CTX - CHANNEL_CTX
    if IS_WINDOWED:
        start_nkv = tl.maximum(
            ((time_beg_block - WINDOW_SIZE) * CHANNEL_CTX) // BLOCK_M - 2, 0
        )
    else:
        start_nkv = 0

    start_nkv = 0
    lo_qk = (start_nkv) * BLOCK_N
    lo_bias = 0 if (REL_BIAS_CHAN or REL_BIAS_TIME) else lo_qk
    if IS_WINDOWED and IS_LAST_QUERY:
        lo_qk = 0
        lo_bias = 0
    lo_qk = tl.multiple_of(lo_qk, BLOCK_N)
    lo_bias = tl.multiple_of(lo_bias, BLOCK_N)
    hi = N_CTX
    if IS_WINDOWED:
        stop_nkv = tl.cdiv(((time_end_block + WINDOW_SIZE + 1) * CHANNEL_CTX), BLOCK_N)
        hi = tl.minimum((stop_nkv) * BLOCK_M, N_CTX)
    if IS_CAUSAL:
        hi = last_time_end_block * BLOCK_M
    hi = tl.multiple_of(hi, BLOCK_M)
    Q += off_qz * stride_qz + off_qh * stride_qh
    Q_bias += off_qh * stride_q_bias_h
    K_glob += off_qz * stride_k_glob_z + off_kh * stride_k_glob_h
    V += off_qz * stride_k_glob_z + off_kh * stride_k_glob_h
    Bias_time += off_qz * stride_bt_z + off_qh * stride_bt_h
    Bias_chan += off_qz * stride_bc_z + off_qh * stride_bc_h
    Out += off_qz * stride_qz + off_qh * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    Q_bias_block_ptr = tl.make_block_ptr(
        base=Q_bias,
        shape=(1, BLOCK_DMODEL),
        strides=(stride_q_bias_h, 1),
        offsets=(0, 0),
        block_shape=(1, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_glob_block_ptr = tl.make_block_ptr(
        base=K_glob,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_k_glob_k, stride_k_glob_n),
        offsets=(0, lo_qk),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(lo_bias, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L,
        shape=(Z * H_Q, N_CTX),
        strides=(N_CTX, 1),
        offsets=(off_hz, start_m * BLOCK_M),
        block_shape=(1, BLOCK_M),
        order=(1, 0),
    )
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0,), eviction_policy="evict_last")
    q_bias = tl.load(Q_bias_block_ptr, eviction_policy="evict_last")
    q = ((q + q_bias) * qk_scale).to(K_glob.dtype.element_ty)
    for start_n in range(lo_bias, lo_qk, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if REL_BIAS_TIME:
            qk += (
                tl.load(
                    Bias_time
                    + offs_m[:, None] * stride_bt_n
                    + TIME_CTX
                    - 1
                    - offs_m[:, None] // CHANNEL_CTX
                    + (start_n + offs_n[None, :]) // CHANNEL_CTX,
                    mask=(start_n + offs_n[None, :]) // CHANNEL_CTX
                    <= offs_m[:, None] // CHANNEL_CTX,
                    eviction_policy="evict_first",
                )
                * qk_scale
            )
        if REL_BIAS_CHAN:
            qk += (
                tl.load(
                    Bias_chan
                    + offs_m[:, None] * stride_bc_n
                    + CHANNEL_CTX
                    - 1
                    - tl.abs(
                        offs_m[:, None] % CHANNEL_CTX
                        - (start_n + offs_n[None, :]) % CHANNEL_CTX
                    ),
                    eviction_policy="evict_first",
                )
                * qk_scale
            )
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        if IS_DROPOUT:
            block_keep_time = (start_n + offs_n[None, :]) // CHANNEL_CTX
            block_keep_chan = (start_n + offs_n[None, :]) % CHANNEL_CTX
            out_keep_time = tl.load(time_dropout + block_keep_time)
            out_keep_chan = tl.load(chan_dropout + block_keep_chan)
            p = tl.where(out_keep_time & out_keep_chan, p / (1 - p_drop), 0.0)
        acc *= alpha[:, None]
        v = tl.load(V_block_ptr, boundary_check=(0,), eviction_policy="evict_first")
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    for start_n in range(lo_qk, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where((start_n + offs_n[None, :]) < N_CTX, qk, float("-inf"))
        k = tl.load(
            K_glob_block_ptr, boundary_check=(1,), eviction_policy="evict_first"
        )
        qk += tl.dot(q, k, allow_tf32=True)
        if IS_WINDOWED and not IS_CAUSAL:
            lq_mask = offs_m[:, None] // CHANNEL_CTX == TIME_CTX - 1
            qk = tl.where(
                (
                    offs_m[:, None] // CHANNEL_CTX
                    >= (start_n + offs_n[None, :]) // CHANNEL_CTX - WINDOW_SIZE
                )
                | (offs_m[:, None] // CHANNEL_CTX == TIME_CTX - 1),
                qk,
                0.0 if REL_BIAS_CHAN or REL_BIAS_TIME else float("-inf"),
            )
            qk = tl.where(
                (
                    offs_m[:, None] // CHANNEL_CTX
                    <= (start_n + offs_n[None, :]) // CHANNEL_CTX + WINDOW_SIZE
                )
                | (offs_m[:, None] // CHANNEL_CTX == TIME_CTX - 1),
                qk,
                0.0 if REL_BIAS_CHAN or REL_BIAS_TIME else float("-inf"),
            )
        if IS_WINDOWED and IS_CAUSAL:
            lq_mask = offs_m[:, None] // CHANNEL_CTX == TIME_CTX - 1
            qk = tl.where(
                (
                    offs_m[:, None] // CHANNEL_CTX
                    <= (start_n + offs_n[None, :]) // CHANNEL_CTX + WINDOW_SIZE
                )
                | (offs_m[:, None] // CHANNEL_CTX == TIME_CTX - 1),
                qk,
                0.0 if REL_BIAS_CHAN or REL_BIAS_TIME else float("-inf"),
            )
        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] // CHANNEL_CTX
                >= (start_n + offs_n[None, :]) // CHANNEL_CTX,
                qk,
                float("-inf"),
            )
        if REL_BIAS_TIME:
            qk += (
                tl.load(
                    Bias_time
                    + offs_m[:, None] * stride_bt_n
                    + TIME_CTX
                    - 1
                    - offs_m[:, None] // CHANNEL_CTX
                    + (start_n + offs_n[None, :]) // CHANNEL_CTX,
                    mask=(start_n + offs_n[None, :]) // CHANNEL_CTX
                    <= offs_m[:, None] // CHANNEL_CTX,
                    eviction_policy="evict_first",
                )
                * qk_scale
            )
        if REL_BIAS_CHAN:
            qk += (
                tl.load(
                    Bias_chan
                    + offs_m[:, None] * stride_bc_n
                    + CHANNEL_CTX
                    - 1
                    - tl.abs(
                        offs_m[:, None] % CHANNEL_CTX
                        - (start_n + offs_n[None, :]) % CHANNEL_CTX
                    ),
                    eviction_policy="evict_first",
                )
                * qk_scale
            )
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_i_new_copy = m_i_new
        m_i_new = tl.where(m_i_new != float("-inf"), m_i_new, 0)
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new_copy
        if IS_DROPOUT:
            block_keep_time = (start_n + offs_n[None, :]) // CHANNEL_CTX
            block_keep_chan = (start_n + offs_n[None, :]) % CHANNEL_CTX
            out_keep_time = tl.load(time_dropout + block_keep_time)
            out_keep_chan = tl.load(chan_dropout + block_keep_chan)
            p = tl.where(out_keep_time & out_keep_chan, p / (1 - p_drop), 0.0)
        acc *= alpha[:, None]
        v = tl.load(V_block_ptr, boundary_check=(0,), eviction_policy="evict_first")
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
        K_glob_block_ptr = tl.advance(K_glob_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    acc = acc / l_i[:, None]
    tl.store(
        L_block_ptr,
        m_i + tl.math.log2(l_i)[None, :],
        boundary_check=(1,),
        cache_modifier=".cs",
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(
        O_block_ptr,
        acc.to(K_glob.dtype.element_ty),
        boundary_check=(0,),
        cache_modifier=".cs",
    )


@jit
def _bwd_kernel_one_col_block(
    Q,
    Q_bias,
    K_glob,
    Bias_time,
    Bias_chan,
    V,
    TCSM,
    CCSM,
    time_dropout,
    chan_dropout,
    sm_scale,
    qk_scale,
    p_drop,
    Out,
    DO,
    DQ,
    DQ_bias,
    DK_glob,
    DBias_time,
    DBias_chan,
    DV,
    L,
    D,
    stride_dqa,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_q_bias_h,
    stride_q_bias_n,
    stride_k_glob_z,
    stride_k_glob_h,
    stride_k_glob_n,
    stride_k_glob_k,
    stride_bt_z,
    stride_bt_h,
    stride_bt_n,
    stride_bt_k,
    stride_bc_z,
    stride_bc_h,
    stride_bc_n,
    stride_bc_k,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    off_hz,
    start_n,
    num_block,
    REL_BIAS_TIME: tl.constexpr,
    REL_BIAS_CHAN: tl.constexpr,
    TIME_CTX: tl.constexpr,
    CHANNEL_CTX: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHAN_PER_BLOCK: tl.constexpr,
    TIME_PER_BLOCK: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    IS_WINDOWED: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    MMA_V3: tl.constexpr,
):
    time_end_block = tl.cdiv((start_n + 1) * BLOCK_N - 1, CHANNEL_CTX)
    time_beg_block = ((start_n) * BLOCK_N) // CHANNEL_CTX
    first_time_beg_block = (time_beg_block * CHANNEL_CTX) // BLOCK_M
    res_time = ((start_n) * BLOCK_M) % CHANNEL_CTX
    if SEQUENCE_PARALLEL:
        DQ += stride_dqa.to(tl.int64) * start_n
    lo = 0
    if IS_WINDOWED:
        start_nkv = tl.maximum(
            ((time_beg_block - WINDOW_SIZE) * CHANNEL_CTX) // BLOCK_M, 0
        )
        lo = start_nkv * BLOCK_M
    if CAUSAL:
        lo = (first_time_beg_block) * BLOCK_M
    lo = tl.multiple_of(lo, BLOCK_M)
    hi_qk = num_block * BLOCK_M
    hi_last = num_block * BLOCK_M
    if IS_WINDOWED:
        stop_nkv = tl.cdiv(((time_end_block + WINDOW_SIZE + 1) * CHANNEL_CTX), BLOCK_N)
        last_query_block = (N_CTX - CHANNEL_CTX) // BLOCK_M
        hi = tl.minimum(last_query_block, stop_nkv + 1)
        hi_qk = hi * BLOCK_M
        hi_last = last_query_block * BLOCK_M
    hi_bias = hi_last if (REL_BIAS_CHAN or REL_BIAS_TIME) else hi_qk
    hi_block = num_block * BLOCK_M
    hi_qk = tl.multiple_of(hi_qk, BLOCK_M)
    hi_last = tl.multiple_of(hi_last, BLOCK_M)
    hi_bias = tl.multiple_of(hi_bias, BLOCK_M)
    hi_block = tl.multiple_of(hi_block, BLOCK_M)
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    Q_bias_block_ptr = tl.make_block_ptr(
        base=Q_bias,
        shape=(1, BLOCK_DMODEL),
        strides=(stride_q_bias_h, 1),
        offsets=(0, 0),
        block_shape=(1, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_glob_block_ptr = tl.make_block_ptr(
        base=K_glob,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_k_glob_n, stride_k_glob_k),
        offsets=(start_n * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(lo, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    if HEAD_RATIO == 1:
        DK_block_ptr = tl.make_block_ptr(
            base=DK_glob,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_k_glob_n, stride_k_glob_k),
            offsets=(start_n * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        DV_block_ptr = tl.make_block_ptr(
            base=DV,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vn, stride_vk),
            offsets=(start_n * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    if REL_BIAS_TIME:
        offs_t = tl.arange(0, CHAN_PER_BLOCK)
    if REL_BIAS_CHAN:
        offs_c = tl.arange(0, TIME_PER_BLOCK)
    L_block_ptr = tl.make_block_ptr(
        base=L,
        shape=(N_CTX, 1),
        strides=(1, N_CTX),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, 1),
        order=(0, 1),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D,
        shape=(N_CTX, 1),
        strides=(1, N_CTX),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, 1),
        order=(0, 1),
    )
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk_glob = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    k_glob = tl.load(
        K_glob_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
        eviction_policy="evict_last",
    )
    v = tl.load(
        V_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
        eviction_policy="evict_last",
    )
    q_bias = tl.load(Q_bias_block_ptr, eviction_policy="evict_last")
    if REL_BIAS_TIME:
        csm_time = tl.load(
            TCSM + (res_time + offs_m)[:, None] * CHAN_PER_BLOCK + offs_t[None, :]
        )
    if REL_BIAS_CHAN:
        csm_chan = tl.load(
            CCSM + (res_time + offs_m)[:, None] * TIME_PER_BLOCK + offs_c[None, :]
        )
    for start_m in range(lo, hi_block, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        q += q_bias
        qk += tl.dot(
            (q * qk_scale).to(K_glob.dtype.element_ty),
            tl.trans(k_glob),
            allow_tf32=True,
        )
        if IS_WINDOWED and not (REL_BIAS_CHAN or REL_BIAS_TIME):
            if CAUSAL:
                qk = tl.where(
                    (
                        (
                            offs_m_curr[:, None] // CHANNEL_CTX
                            >= (offs_n[None, :]) // CHANNEL_CTX
                        )
                        & (
                            offs_m_curr[:, None] // CHANNEL_CTX
                            <= (offs_n[None, :]) // CHANNEL_CTX + WINDOW_SIZE
                        )
                    )
                    | (offs_m_curr[:, None] // CHANNEL_CTX == TIME_CTX - 1),
                    qk,
                    float("-inf"),
                )
            else:
                qk = tl.where(
                    (
                        (
                            offs_m_curr[:, None] // CHANNEL_CTX
                            >= (offs_n[None, :]) // CHANNEL_CTX - WINDOW_SIZE
                        )
                        & (
                            offs_m_curr[:, None] // CHANNEL_CTX
                            <= (offs_n[None, :]) // CHANNEL_CTX + WINDOW_SIZE
                        )
                    )
                    | (offs_m_curr[:, None] // CHANNEL_CTX == TIME_CTX - 1),
                    qk,
                    float("-inf"),
                )
        if IS_WINDOWED and (REL_BIAS_CHAN or REL_BIAS_TIME):
            qk = tl.where(
                (
                    offs_m_curr[:, None] // CHANNEL_CTX
                    <= (offs_n[None, :]) // CHANNEL_CTX + WINDOW_SIZE
                )
                | (offs_m_curr[:, None] // CHANNEL_CTX == TIME_CTX - 1),
                qk,
                0,
            )
            qk = tl.where(
                offs_m_curr[:, None] // CHANNEL_CTX >= (offs_n[None, :]) // CHANNEL_CTX,
                qk,
                float("-inf"),
            )
        if not IS_WINDOWED and CAUSAL:
            qk = tl.where(
                offs_m_curr[:, None] // CHANNEL_CTX >= (offs_n[None, :]) // CHANNEL_CTX,
                qk,
                float("-inf"),
            )
        if REL_BIAS_TIME:
            qk += (
                tl.load(
                    Bias_time
                    + offs_m_curr[:, None] * stride_bt_n
                    + TIME_CTX
                    - 1
                    - offs_m_curr[:, None] // CHANNEL_CTX
                    + (offs_n[None, :]) // CHANNEL_CTX,
                    mask=(offs_n[None, :]) // CHANNEL_CTX
                    <= offs_m_curr[:, None] // CHANNEL_CTX,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                * qk_scale
            )
        if REL_BIAS_CHAN:
            qk += (
                tl.load(
                    Bias_chan
                    + offs_m_curr[:, None] * stride_bc_n
                    + CHANNEL_CTX
                    - 1
                    - tl.abs(
                        offs_m_curr[:, None] % CHANNEL_CTX
                        - (offs_n[None, :]) % CHANNEL_CTX
                    ),
                    eviction_policy="evict_first",
                )
                * qk_scale
            )
        qk = tl.where(
            (offs_n[None, :] < N_CTX) & (offs_m_curr[:, None] < N_CTX),
            qk,
            float("-inf"),
        )
        l_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        p = tl.math.exp2(qk - l_i)
        if IS_DROPOUT:
            block_keep_time = (offs_n[None, :]) // CHANNEL_CTX
            block_keep_chan = (offs_n[None, :]) % CHANNEL_CTX
            out_keep_time = tl.load(time_dropout + block_keep_time)
            out_keep_chan = tl.load(chan_dropout + block_keep_chan)
            out_keep = out_keep_time & out_keep_chan
            p_dr = tl.where(out_keep, p / (1 - p_drop), 0.0)
        do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
        if IS_DROPOUT:
            dv += tl.dot(tl.trans(p_dr.to(Q.dtype.element_ty)), do, allow_tf32=True)
        else:
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, allow_tf32=True)
        Di = tl.load(D_block_ptr, boundary_check=(0,))
        dp = tl.dot(do, tl.trans(v), allow_tf32=True)
        if IS_DROPOUT:
            dp = tl.where(out_keep, dp / (1 - p_drop), 0.0)
        ds = (p * (dp - Di) * sm_scale).to(Q.dtype.element_ty)
        if REL_BIAS_TIME:
            bt = tl.dot(ds, csm_time, allow_tf32=True)
            tl.atomic_add(
                DBias_time
                + offs_m_curr[:, None] * stride_bt_n
                + TIME_CTX
                - 1
                - offs_m_curr[:, None] // CHANNEL_CTX
                + (time_beg_block + offs_t)[None, :],
                bt,
                mask=(time_beg_block + offs_t)[None, :]
                <= offs_m_curr[:, None] // CHANNEL_CTX,
                sem="relaxed",
            )
        if REL_BIAS_CHAN:
            bc = tl.dot(ds, csm_chan, allow_tf32=True)
            tl.atomic_add(
                DBias_chan
                + offs_m_curr[:, None] * stride_bc_n
                + CHANNEL_CTX
                - 1
                - tl.abs(
                    offs_m_curr[:, None] % CHANNEL_CTX - (offs_c[None, :]) % CHANNEL_CTX
                ),
                bc,
                sem="relaxed",
            )
        if IS_WINDOWED and (REL_BIAS_CHAN or REL_BIAS_TIME):
            ds = tl.where(
                (
                    offs_m_curr[:, None] // CHANNEL_CTX
                    <= (offs_n[None, :]) // CHANNEL_CTX + WINDOW_SIZE
                )
                | (offs_m_curr[:, None] // CHANNEL_CTX == TIME_CTX - 1),
                ds,
                0,
            )
        dk_glob += tl.dot(tl.trans(ds).to(Q.dtype.element_ty), q, allow_tf32=True)
        if not SEQUENCE_PARALLEL:
            dq = tl.load(
                DQ_block_ptr, boundary_check=(0,), eviction_policy="evict_last"
            )
            new_dq = tl.dot(ds.to(Q.dtype.element_ty), k_glob, allow_tf32=True)
            dq += new_dq
            tl.store(
                DQ_block_ptr,
                dq,
                boundary_check=(0,),
                eviction_policy="evict_last",
            )
        elif SEQUENCE_PARALLEL:
            if MMA_V3:
                new_dq = tl.dot(ds, k_glob, allow_tf32=True)
            else:
                new_dq = tl.trans(
                    tl.dot(tl.trans(k_glob), tl.trans(ds), allow_tf32=True)
                )
            tl.store(
                DQ_block_ptr,
                new_dq,
                boundary_check=(0,),
            )
        new_dq = tl.where((offs_m_curr[:, None]) < N_CTX, new_dq, 0)
        tl.atomic_add(
            DQ_bias + offs_k[None, :], tl.sum(new_dq, 0)[None, :], sem="relaxed"
        )
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
        L_block_ptr = tl.advance(L_block_ptr, (BLOCK_M, 0))
        D_block_ptr = tl.advance(D_block_ptr, (BLOCK_M, 0))
    if HEAD_RATIO == 1:
        tl.store(DV_block_ptr, dv.to(DV.dtype.element_ty), boundary_check=(0,))
        tl.store(
            DK_block_ptr, dk_glob.to(DK_glob.dtype.element_ty), boundary_check=(0,)
        )
    else:
        dv_ptrs = DV + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        dk_glob_ptrs = DK_glob + (
            offs_n[:, None] * stride_k_glob_n + offs_k[None, :] * stride_k_glob_k
        )
        tl.atomic_add(
            dv_ptrs,
            dv.to(DV.dtype.element_ty),
            mask=offs_n[:, None] < N_CTX,
            sem="relaxed",
        )
        tl.atomic_add(
            dk_glob_ptrs,
            dk_glob.to(DK_glob.dtype.element_ty),
            mask=offs_n[:, None] < N_CTX,
            sem="relaxed",
        )


@jit
def _bwd_kernel(
    Q,
    Q_bias,
    K_glob,
    Bias_time,
    Bias_chan,
    V,
    TCSM,
    CCSM,
    time_dropout,
    chan_dropout,
    sm_scale,
    p_drop,
    Out,
    DO,
    DQ,
    DQ_bias,
    DK_glob,
    DBias_time,
    DBias_chan,
    DV,
    L,
    D,
    stride_dqa,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_q_bias_h,
    stride_q_bias_n,
    stride_k_glob_z,
    stride_k_glob_h,
    stride_k_glob_n,
    stride_k_glob_k,
    stride_bt_z,
    stride_bt_h,
    stride_bt_n,
    stride_bt_k,
    stride_bc_z,
    stride_bc_h,
    stride_bc_n,
    stride_bc_k,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    Z: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    N_CTX: tl.constexpr,
    REL_BIAS_TIME: tl.constexpr,
    REL_BIAS_CHAN: tl.constexpr,
    TIME_CTX: tl.constexpr,
    CHANNEL_CTX: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHAN_PER_BLOCK: tl.constexpr,
    TIME_PER_BLOCK: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    IS_WINDOWED: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    MMA_V3: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    off_hz = tl.program_id(0)
    off_qz = off_hz // H_Q
    off_qh = off_hz % H_Q
    off_kh = (off_hz // HEAD_RATIO) % H_KV
    Q += off_qz * stride_qz + off_qh * stride_qh
    Q_bias += off_qh * stride_q_bias_h
    K_glob += off_qz * stride_k_glob_z + off_kh * stride_k_glob_h
    V += off_qz * stride_vz + off_kh * stride_vh
    DO += off_qz * stride_qz + off_qh * stride_qh
    DQ += off_qz * stride_qz + off_qh * stride_qh
    DQ_bias += off_qh * stride_q_bias_h
    DK_glob += off_qz * stride_k_glob_z + off_kh * stride_k_glob_h
    DV += off_qz * stride_vz + off_kh * stride_vh
    L += off_hz * N_CTX
    D += off_hz * N_CTX
    if REL_BIAS_TIME:
        Bias_time += off_qz * stride_bt_z + off_qh * stride_bt_h
        DBias_time += off_qz * stride_bt_z + off_qh * stride_bt_h
    if REL_BIAS_CHAN:
        Bias_chan += off_qz * stride_bc_z + off_qh * stride_bc_h
        DBias_chan += off_qz * stride_bc_z + off_qh * stride_bc_h
    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q,
                Q_bias,
                K_glob,
                Bias_time,
                Bias_chan,
                V,
                TCSM,
                CCSM,
                time_dropout,
                chan_dropout,
                sm_scale,
                qk_scale,
                p_drop,
                Out,
                DO,
                DQ,
                DQ_bias,
                DK_glob,
                DBias_time,
                DBias_chan,
                DV,
                L,
                D,
                stride_dqa,
                stride_qz,
                stride_qh,
                stride_qm,
                stride_qk,
                stride_q_bias_h,
                stride_q_bias_n,
                stride_k_glob_z,
                stride_k_glob_h,
                stride_k_glob_n,
                stride_k_glob_k,
                stride_bt_z,
                stride_bt_h,
                stride_bt_n,
                stride_bt_k,
                stride_bc_z,
                stride_bc_h,
                stride_bc_n,
                stride_bc_k,
                stride_vz,
                stride_vh,
                stride_vn,
                stride_vk,
                Z,
                H_Q,
                N_CTX,
                off_hz,
                start_n,
                num_block_n,
                REL_BIAS_TIME=REL_BIAS_TIME,
                REL_BIAS_CHAN=REL_BIAS_CHAN,
                TIME_CTX=TIME_CTX,
                CHANNEL_CTX=CHANNEL_CTX,
                HEAD_RATIO=HEAD_RATIO,
                WINDOW_SIZE=WINDOW_SIZE,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                CHAN_PER_BLOCK=CHAN_PER_BLOCK,
                TIME_PER_BLOCK=TIME_PER_BLOCK,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                IS_WINDOWED=IS_WINDOWED,
                IS_DROPOUT=IS_DROPOUT,
                MMA_V3=MMA_V3,
            )
    else:
        start_n = tl.program_id(1)
        _bwd_kernel_one_col_block(
            Q,
            Q_bias,
            K_glob,
            Bias_time,
            Bias_chan,
            V,
            TCSM,
            CCSM,
            time_dropout,
            chan_dropout,
            sm_scale,
            qk_scale,
            p_drop,
            Out,
            DO,
            DQ,
            DQ_bias,
            DK_glob,
            DBias_time,
            DBias_chan,
            DV,
            L,
            D,
            stride_dqa,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_q_bias_h,
            stride_q_bias_n,
            stride_k_glob_z,
            stride_k_glob_h,
            stride_k_glob_n,
            stride_k_glob_k,
            stride_bt_z,
            stride_bt_h,
            stride_bt_n,
            stride_bt_k,
            stride_bc_z,
            stride_bc_h,
            stride_bc_n,
            stride_bc_k,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            Z,
            H_Q,
            N_CTX,
            off_hz,
            start_n,
            num_block_n,
            REL_BIAS_TIME=REL_BIAS_TIME,
            REL_BIAS_CHAN=REL_BIAS_CHAN,
            TIME_CTX=TIME_CTX,
            CHANNEL_CTX=CHANNEL_CTX,
            HEAD_RATIO=HEAD_RATIO,
            WINDOW_SIZE=WINDOW_SIZE,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            CHAN_PER_BLOCK=CHAN_PER_BLOCK,
            TIME_PER_BLOCK=TIME_PER_BLOCK,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            IS_WINDOWED=IS_WINDOWED,
            IS_DROPOUT=IS_DROPOUT,
            MMA_V3=MMA_V3,
        )


class FlashMVPA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        q_bias,
        k_glob,
        bias_time,
        bias_chan,
        v,
        time_ctx,
        chan_ctx,
        causal,
        sm_scale,
        window_size=0,
        p_drop=0.0,
        sequence_parallel=False,
        rel_bias_time=True,
        rel_bias_chan=True,
        generator=None,
    ):
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Flash attention currently only supported for compute capability >= 80"
            )
        BLOCK_M = 128
        BLOCK_N = 64
        Lq, Lk, Lv = q.shape[-1], k_glob.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        assert q.shape[1] % k_glob.shape[1] == 0
        assert time_ctx > 0 and chan_ctx > 0
        head_ratio = q.shape[1] // k_glob.shape[1]
        o = torch.empty_like(q)
        grid = (cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        if p_drop > 0:
            p_per_dim = 1 - sqrt(1 - p_drop)
            chan_dropout = (
                torch.rand(chan_ctx, device="cuda", generator=generator) > p_per_dim
            )
            time_dropout = (
                torch.rand(time_ctx, device="cuda", generator=generator) > p_per_dim
            )
        else:
            chan_dropout = torch.empty(0)
            time_dropout = torch.empty(0)
        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q,
            q_bias,
            k_glob,
            bias_time,
            bias_chan,
            v,
            time_dropout if p_drop > 0 else None,
            chan_dropout if p_drop > 0 else None,
            sm_scale,
            p_drop,
            L,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q_bias.stride(0),
            q_bias.stride(1),
            k_glob.stride(0),
            k_glob.stride(1),
            k_glob.stride(2),
            k_glob.stride(3),
            bias_time.stride(0),
            bias_time.stride(1),
            bias_time.stride(2),
            bias_time.stride(3),
            bias_chan.stride(0),
            bias_chan.stride(1),
            bias_chan.stride(2),
            bias_chan.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            k_glob.shape[1],
            q.shape[2],
            k_glob.shape[0] * k_glob.shape[1] * k_glob.shape[2],
            q.shape[0] * q.shape[1] * q.shape[2],
            REL_BIAS_TIME=rel_bias_time,
            REL_BIAS_CHAN=rel_bias_chan,
            TIME_CTX=time_ctx,
            CHANNEL_CTX=chan_ctx,
            WINDOW_SIZE=window_size,
            HEAD_RATIO=head_ratio,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            IS_WINDOWED=True if window_size > 0 else False,
            IS_DROPOUT=True if p_drop > 0 else False,
            num_warps=num_warps,
            num_stages=4,
        )
        ctx.save_for_backward(
            q, q_bias, k_glob, bias_time, bias_chan, v, time_dropout, chan_dropout, o, L
        )
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.sequence_parallel = sequence_parallel
        ctx.window_size = window_size
        ctx.p_drop = p_drop
        ctx.time_ctx = time_ctx
        ctx.chan_ctx = chan_ctx
        ctx.rel_bias_time = rel_bias_time
        ctx.rel_bias_chan = rel_bias_chan
        return o

    @staticmethod
    def backward(ctx, do):
        def next_power_of_2(x):
            nx = 1 << (x - 1).bit_length()
            if nx < 16:
                nx = 16
            return nx

        capability = torch.cuda.get_device_capability()
        MMA_V3 = capability[0] >= 9
        BLOCK = 128 if ctx.chan_ctx > 16 else 64
        (
            q,
            q_bias,
            k_glob,
            bias_time,
            bias_chan,
            v,
            time_dropout,
            chan_dropout,
            o,
            L,
        ) = ctx.saved_tensors
        assert q.shape[1] % k_glob.shape[1] == 0
        head_ratio = q.shape[1] // k_glob.shape[1]
        sequence_parallel = ctx.sequence_parallel
        seq_len_kv = k_glob.shape[2]
        do = do.contiguous()
        if sequence_parallel:
            replicas = cdiv(seq_len_kv, BLOCK)
            new_dq_shape = (replicas,) + q.shape
            dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
        else:
            dq = torch.zeros_like(q, dtype=torch.float32)
        dk_glob = torch.zeros_like(k_glob, dtype=torch.float32)
        dq_bias = torch.zeros_like(q_bias, dtype=torch.float32)
        dbias_time = torch.zeros_like(bias_time, dtype=torch.float32)
        dbias_chan = torch.zeros_like(bias_chan, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        if ctx.rel_bias_time:
            time_block = torch.ones((ctx.chan_ctx, 1), device="cuda")
            time_full_blocks = time_block.unsqueeze(0).repeat(
                cdiv(BLOCK, ctx.chan_ctx), 1, 1
            )
            time_last_block = time_block[: (BLOCK + ctx.chan_ctx - 1) % ctx.chan_ctx]
            max_times_in_block = cdiv(BLOCK + ctx.chan_ctx - 1, ctx.chan_ctx)
            time_csm = torch.zeros(
                BLOCK + ctx.chan_ctx - 1,
                next_power_of_2(max_times_in_block),
                device="cuda",
                dtype=q.dtype,
            )
            time_csm[:, :max_times_in_block] = torch.block_diag(
                *time_full_blocks,
                (
                    time_last_block
                    if time_last_block.shape[0] > 0
                    else torch.empty(0, 0, device="cuda")
                ),
            )
        if ctx.rel_bias_chan:
            chan_len = BLOCK + ctx.chan_ctx - 1
            chan_block = torch.diag(torch.ones(ctx.chan_ctx, device="cuda"))
            chan_full_blocks = chan_block.unsqueeze(0).repeat(
                cdiv(chan_len, ctx.chan_ctx), 1, 1
            )
            chan_csm = torch.zeros(
                chan_len,
                next_power_of_2(ctx.chan_ctx),
                device="cuda",
                dtype=q.dtype,
            )
            chan_csm[:, : ctx.chan_ctx] = chan_full_blocks.flatten(0, 1)[:chan_len]
        delta = torch.sum(do * o, dim=-1).flatten(0, 1)
        _bwd_kernel[(ctx.grid[1], cdiv(seq_len_kv, BLOCK) if sequence_parallel else 1)](
            q,
            q_bias,
            k_glob,
            bias_time,
            bias_chan,
            v,
            time_csm if ctx.rel_bias_time else None,
            chan_csm if ctx.rel_bias_chan else None,
            time_dropout if ctx.p_drop > 0 else None,
            chan_dropout if ctx.p_drop > 0 else None,
            ctx.sm_scale,
            ctx.p_drop,
            o,
            do,
            dq,
            dq_bias,
            dk_glob,
            dbias_time,
            dbias_chan,
            dv,
            L,
            delta,
            o.numel(),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q_bias.stride(0),
            q_bias.stride(1),
            k_glob.stride(0),
            k_glob.stride(1),
            k_glob.stride(2),
            k_glob.stride(3),
            bias_time.stride(0),
            bias_time.stride(1),
            bias_time.stride(2),
            bias_time.stride(3),
            bias_chan.stride(0),
            bias_chan.stride(1),
            bias_chan.stride(2),
            bias_chan.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            q.shape[0],
            q.shape[1],
            k_glob.shape[1],
            q.shape[2],
            REL_BIAS_TIME=ctx.rel_bias_time,
            REL_BIAS_CHAN=ctx.rel_bias_chan,
            TIME_CTX=ctx.time_ctx,
            CHANNEL_CTX=ctx.chan_ctx,
            HEAD_RATIO=head_ratio,
            WINDOW_SIZE=ctx.window_size,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            CHAN_PER_BLOCK=time_csm.shape[1] if ctx.rel_bias_time else 0,
            TIME_PER_BLOCK=chan_csm.shape[1] if ctx.rel_bias_chan else 0,
            SEQUENCE_PARALLEL=sequence_parallel,
            CAUSAL=ctx.causal,
            IS_WINDOWED=True if ctx.window_size > 0 else False,
            IS_DROPOUT=True if ctx.p_drop > 0 else False,
            MMA_V3=MMA_V3,
            num_warps=8,
            num_stages=1,
        )
        if len(dq.shape) == 5:
            dq = dq.sum(dim=0)
        return (
            dq,
            dq_bias,
            dk_glob,
            dbias_time,
            dbias_chan,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _rel_bias(q, q_btime, q_bchan, k_time, k_chan, q_kv_ratio: int = 1):
    k_time = repeat_kv(k_time, q_kv_ratio)
    k_chan = repeat_kv(k_chan, q_kv_ratio)
    q_btime = repeat_kv(q_btime, q_kv_ratio)
    q_bchan = repeat_kv(q_bchan, q_kv_ratio)
    q_time = q + q_btime
    q_chan = q + q_bchan
    time_att = torch.matmul(q_time, k_time.transpose(-1, -2))
    channel_att = torch.matmul(q_chan, k_chan.transpose(-1, -2))
    return time_att, channel_att


def flash_mvpa(
    q,
    q_bias,
    q_btime,
    q_bchan,
    k_glob,
    k_time,
    k_chan,
    v,
    causal,
    sm_scale,
    window_size=0,
    p_drop=0.0,
    sequence_parallel=False,
    rel_bias_time=True,
    rel_bias_chan=True,
    generator=None,
):
    if rel_bias_chan or rel_bias_time:
        causal = True
    q_heads = q.shape[1]
    kv_heads = v.shape[1]
    assert q_heads % kv_heads == 0, "Q heads must be divisible by KV heads."
    q_kv_ratio = int(q_heads / kv_heads)
    n_ctx = q.shape[2]
    time_ctx = k_time.shape[2]
    chan_ctx = k_chan.shape[2]
    assert n_ctx == time_ctx * chan_ctx
    time_att, channel_att = _rel_bias(q, q_btime, q_bchan, k_time, k_chan, q_kv_ratio)
    q_bias = repeat_kv(q_bias, q_kv_ratio)
    tfa = FlashMVPA.apply
    out = tfa(
        q,
        q_bias.squeeze(),
        k_glob,
        time_att,
        channel_att,
        v,
        time_ctx,
        chan_ctx,
        causal,
        sm_scale,
        window_size,
        p_drop,
        sequence_parallel,
        rel_bias_time,
        rel_bias_chan,
        generator,
    )
    return out
