"""Tracked reference kernels from the September 12 CA-overlap backward profile.

The bodies below reproduce the exact profiled arithmetic/indexing at shape
B=4, chunks=16, heads=32, chunk_size=128, head_dim=64. They are kept as source,
not imported from generated compiler caches. Names 17/18 in that trace consumed
31.907/77.576 ms per rank-0 update. Their measured launch choices were
XBLOCK=64/YBLOCK=16/4 warps and XBLOCK=64/YBLOCK=64/8 warps respectively.

Provenance: ops/model sources and profile-ca-overlap trace map these bodies to
cache hashes cq665jkyfy6uac7wrzvwa5aigwids6rzt4n7vzip6xvufai4ajjw and
cbr6gfx3vjhnzjbykfch3pg5peunfivhgdrfvl5hnzolhvfjfgdy. The isolated launcher
preallocates both scratch tensors; production reused a dead buffer for one of
them. Compare kernel time and traffic, not an inferred peak-allocation saving.
"""

import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers


@triton.jit
def _profile_rotary(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp33 = tl.load(in_ptr0 + (x2 + 64*y3), xmask & ymask).to(tl.float32)
    tmp34 = tl.load(in_ptr1 + (x2 + 64*y3), xmask & ymask).to(tl.float32)
    tmp36 = tl.load(in_ptr3 + (x2 + 64*y0), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (x2).to(tl.int64)
    tmp1 = (tmp0).to(tl.int64)
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = ((((-1) + x2) % 2)).to(tl.int64)
    tmp5 = (tmp4).to(tl.int64)
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp3 & tmp7
    tmp9 = tl.load(in_ptr0 + (2*(triton_helpers.div_floor_integer((-1) + x2,  2)) + 64*y3), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (2*(triton_helpers.div_floor_integer((-1) + x2,  2)) + 64*y3), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr2 + (2*(triton_helpers.div_floor_integer((-1) + x2,  2)) + 64*y0), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 * tmp13
    tmp15 = -tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.full([1, 1], 0.0, tl.float32)
    tmp19 = tl.where(tmp8, tmp17, tmp18)
    tmp20 = ((x2 % 2)).to(tl.int64)
    tmp21 = (tmp20).to(tl.int64)
    tmp22 = tmp21 == tmp6
    tmp23 = tl.load(in_ptr0 + (1 + 2*(x2 // 2) + 64*y3), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (1 + 2*(x2 // 2) + 64*y3), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr2 + (1 + 2*(x2 // 2) + 64*y0), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 * tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp22, tmp28, tmp29)
    tmp31 = tl.where(tmp22, tmp30, tmp18)
    tmp32 = tmp19 + tmp31
    tmp35 = tmp33 + tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 * tmp37
    tmp39 = tmp32 + tmp38
    tmp40 = tl.load(in_ptr4 + (y0 + 256*(triton_helpers.div_floor_integer((-1) + x2,  2)) + 8192*y1), tmp8 & xmask & ymask, other=0.0).to(tl.float32)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tl.load(in_ptr5 + (2*(triton_helpers.div_floor_integer((-1) + x2,  2)) + 64*y3), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp43 * tmp43
    tmp45 = tl.full([1, 1], 1.0, tl.float32)
    tmp46 = tmp45 - tmp44
    tmp47 = tmp41 * tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.load(in_ptr6 + (2*(triton_helpers.div_floor_integer((-1) + x2,  2)) + 64*y3), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tmp48 + tmp49
    tmp51 = tmp50 * tmp13
    tmp52 = -tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp8, tmp52, tmp53)
    tmp55 = tl.where(tmp8, tmp54, tmp18)
    tmp56 = tl.load(in_ptr4 + (128 + y0 + 256*(x2 // 2) + 8192*y1), tmp22 & xmask & ymask, other=0.0).to(tl.float32)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tl.load(in_ptr5 + (1 + 2*(x2 // 2) + 64*y3), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp59 * tmp59
    tmp61 = tl.full([1, 1], 1.0, tl.float32)
    tmp62 = tmp61 - tmp60
    tmp63 = tmp57 * tmp62
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tl.load(in_ptr6 + (1 + 2*(x2 // 2) + 64*y3), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp66 = tmp64 + tmp65
    tmp67 = tmp66 * tmp27
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp22, tmp67, tmp68)
    tmp70 = tl.where(tmp22, tmp69, tmp18)
    tmp71 = tmp55 + tmp70
    tl.store(out_ptr0 + (x2 + 64*y3), tmp39, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 64*y3), tmp71, xmask & ymask)


@triton.jit
def _profile_pack(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 10240
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    y3 = yindex
    tmp0 = (x2).to(tl.int32)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x2).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1, 1], 2048, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (64*((y0 % 128)) + 8192*((((x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + (((x2) % 64))), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp4 >= tmp5
    tmp9 = tl.full([1, 1], 4096, tl.int64)
    tmp10 = tmp4 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + (64*((y0 % 128)) + 8192*(((((-2048) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-2048) + x2) % 64))), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (128*((((-2048) + x2) % 2048)) + 262144*(y0 // 128) + 4194304*y1 + ((y0 % 128))), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (64*((y0 % 128)) + 8192*(((((-2048) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-2048) + x2) % 64))), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp16
    tmp18 = tl.full([1, 1], 1.0, tl.float32)
    tmp19 = tmp18 - tmp17
    tmp20 = tmp14 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr4 + (64*((y0 % 128)) + 8192*(((((-2048) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-2048) + x2) % 64))), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.load(in_ptr5 + (64*((y0 % 128)) + ((((-2048) + x2) % 64))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 * tmp25
    tmp27 = tmp12 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp11, tmp27, tmp28)
    tmp30 = tmp4 >= tmp9
    tmp31 = tl.full([1, 1], 6144, tl.int64)
    tmp32 = tmp4 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr6 + (64*((y0 % 128)) + 8192*(((((-4096) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-4096) + x2) % 64))), tmp33 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tmp4 >= tmp31
    tmp36 = tl.full([1, 1], 8192, tl.int64)
    tmp37 = tmp4 < tmp36
    tmp38 = tmp35 & tmp37
    tmp39 = tl.load(in_ptr7 + (2048*y3 + ((((-6144) + x2) % 2048))), tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr8 + (64*((y0 % 128)) + 8192*(((((-6144) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-6144) + x2) % 64))), tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tl.full([1, 1], 0.125, tl.float32)
    tmp42 = tmp40 * tmp41
    tmp43 = tmp39 * tmp42
    tmp44 = -tmp43
    tmp45 = tl.load(in_ptr9 + (64*((y0 % 128)) + 8192*(((((-6144) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-6144) + x2) % 64))), tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp46 = tmp39 * tmp45
    tmp47 = tmp44 + tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.load(in_ptr10 + (64*((y0 % 128)) + 8192*(((((-6144) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-6144) + x2) % 64))), tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.full([1, 1], 1.0, tl.float32)
    tmp52 = tmp51 - tmp50
    tmp53 = tmp50 * tmp52
    tmp54 = tmp48 * tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp38, tmp55, tmp56)
    tmp58 = tmp0 >= tmp36
    tmp59 = tl.full([1, 1], 10240, tl.int64)
    tmp60 = tmp0 < tmp59
    tmp61 = tl.load(in_ptr11 + (64*((y0 % 128)) + 8192*(((((-8192) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-8192) + x2) % 64))), tmp58 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tl.load(in_ptr12 + (64*((y0 % 128)) + 8192*(((((-8192) + x2) // 64) % 32)) + 262144*(y0 // 128) + 4194304*y1 + ((((-8192) + x2) % 64))), tmp58 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64 * tmp64
    tmp66 = tl.full([1, 1], 1.0, tl.float32)
    tmp67 = tmp66 - tmp65
    tmp68 = tmp62 * tmp67
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp58, tmp69, tmp70)
    tmp72 = tl.where(tmp38, tmp57, tmp71)
    tmp73 = tl.where(tmp33, tmp34, tmp72)
    tmp74 = tl.where(tmp11, tmp29, tmp73)
    tmp75 = tl.where(tmp6, tmp7, tmp74)
    tl.store(out_ptr0 + (x2 + 10240*y3), tmp75, xmask)
