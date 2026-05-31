# Minimum independent terms per Julia thread before using @spawn parallel path.
const MIN_TERMS_PER_THREAD = 2
# Minimum fusion paths per thread inside `apply_twosides!` (block2 Quanta-style; only when term matvec is serial).
const MIN_FUSION_PATHS_PER_THREAD = 4

struct TwosideFusionPath
	f1l::Any
	f1r::Any
	adjoint::Bool
end

struct TwoSideTerm{A}
    left::A
    right::A
    coeff::Float64
    add_adjoint::Bool
end

mutable struct QCCenter{A, B, T<:Number}
	Hleft::B
	Hright::B
	adagTleft::Vector{A}
	adagTright::Vector{A}
	Tdagaleft::Vector{A}
	Tdagaright::Vector{A}
	PAleft::Matrix{A}
	PAright::Matrix{A}
	BQleft::Matrix{A}
	BQright::Matrix{A}
	workspace::Vector{T}
	terms::Vector{TwoSideTerm{A}}
	thread_ys::Union{Nothing, Vector{B}}
	thread_workspaces::Union{Nothing, Vector{Vector{T}}}
	diag_aa::Union{Nothing, Vector{Float64}}
end

function _operator_eltype(s::QCSiteStorages)
	for i in eachindex(s.adagT)
		isassigned(s.adagT, i) && return typeof(s.adagT[i])
	end
	for i in eachindex(s.Tdaga)
		isassigned(s.Tdaga, i) && return typeof(s.Tdaga[i])
	end
	for i in eachindex(s.PA)
		isassigned(s.PA, i) && return typeof(s.PA[i])
	end
	for i in eachindex(s.BQ)
		isassigned(s.BQ, i) && return typeof(s.BQ[i])
	end
	error("QCSiteStorages has no assigned off-diagonal operators")
end

function QCCenter(left::QCSiteStorages, right::QCSiteStorages)
	adagTleft, adagTright = filter_pair_vector(left.adagT, right.adagT)
	Tdagaleft, Tdagaright = filter_pair_vector(left.Tdaga, right.Tdaga)
	Hleft, Hright = left.H, right.H
	n = max_blocksize(left.H)
	n = max(max_blocksize(right.H), n)
	workspace = zeros(scalartype(Hleft), n * n)
	A = _operator_eltype(left)
	terms = TwoSideTerm{A}[]
	res = QCCenter(
		left.H, right.H,
		adagTleft, adagTright, Tdagaleft, Tdagaright,
		left.PA, right.PA, left.BQ, right.BQ,
		workspace, terms,
		nothing, nothing,
		nothing,
	)
	append!(res.terms, _build_twoside_terms(res))
	return res
end

function _build_twoside_terms(m::QCCenter{A,B,T}) where {A,B,T}
    terms = TwoSideTerm{A}[]

    for (l, r) in zip(m.adagTleft, m.adagTright)
        push!(terms, TwoSideTerm(l, r, 1.0, true))
    end

    for (l, r) in zip(m.Tdagaleft, m.Tdagaright)
        push!(terms, TwoSideTerm(l, r, 1.0, true))
    end

    for r in 1:size(m.PAleft, 1)
        for s in (r + 1):size(m.PAleft, 2)
            if isassigned(m.PAleft, r, s) && isassigned(m.PAright, r, s)
                push!(terms, TwoSideTerm(m.PAleft[r, s], m.PAright[r, s], 1.0, true))
            end
        end
    end

    for r in 1:size(m.BQleft, 1)
        if isassigned(m.BQleft, r, r) && isassigned(m.BQright, r, r)
            push!(terms, TwoSideTerm(m.BQleft[r, r], m.BQright[r, r], -1.0, false))
        end
        for s in (r + 1):size(m.BQleft, 2)
            if isassigned(m.BQleft, r, s) && isassigned(m.BQright, r, s)
                push!(terms, TwoSideTerm(m.BQleft[r, s], m.BQright[r, s], -1.0, true))
            end
        end
    end

    return terms
end

function _ensure_thread_buffers!(m::QCCenter{A,B,T}, y::B) where {A,B,T}
	if m.thread_ys === nothing
		nt = Threads.nthreads()
		m.thread_ys = Vector{B}(undef, nt)
		m.thread_workspaces = Vector{Vector{T}}(undef, nt)
		for tid in 1:nt
			m.thread_ys[tid] = similar(y)
			m.thread_workspaces[tid] = similar(m.workspace)
		end
	end
	return m.thread_ys, m.thread_workspaces
end

max_blocksize(x::AbstractTensorMap) = mapreduce(a -> size(a[2], 1), max, blocks(x); init = 0)

function calc_galerkin(m::QCCenter, x::MPSBondTensor)
	out = m(x)
	try
		return norm(leftnull(x)' * out)
	catch
		return norm(out * rightnull(x)')
	end
end

function _mul_twoside_terms_serial!(y, m::QCCenter, x)
	for t in m.terms
		apply_twosides!(
			y, x, t.left, t.right, m.workspace, t.coeff;
			add_adjoint = t.add_adjoint,
		)
	end
	return y
end

function _mul_twoside_terms_parallel!(y, m::QCCenter, x)
	ys, workspaces = _ensure_thread_buffers!(m, y)
	nt = length(ys)
	for tid in 1:nt
		fill!(ys[tid], 0)
	end

	terms = m.terms
	nexti = Threads.Atomic{Int}(1)
	tasks = Vector{Task}(undef, nt)
	for tid in 1:nt
		tasks[tid] = Threads.@spawn begin
			yt = ys[tid]
			wt = workspaces[tid]
			while true
				i = Threads.atomic_add!(nexti, 1)
				i > length(terms) && break
				t = terms[i]
				apply_twosides!(
					yt, x, t.left, t.right, wt, t.coeff;
					add_adjoint = t.add_adjoint,
					parallel_fusion = false,
				)
			end
		end
	end
	wait.(tasks)

	for yt in ys
		axpy!(true, yt, y)
	end
	return y
end

function _use_parallel_terms(m::QCCenter)
	nt = Threads.nthreads()
	return nt > 1 &&
		!isempty(m.terms) &&
		length(m.terms) >= nt * MIN_TERMS_PER_THREAD
end

function _flat_sector_offsets(template::TensorMap)
	offs = Dict{Any, Int}()
	idx = 1
	for (key, blk) in blocks(template)
		offs[key] = idx
		idx += length(blk)
	end
	return offs
end

"""``⟨e_{r,c}| H_left | e_{r,c}⟩ = H_left[r,r]``, ``⟨e_{r,c}| H_right | e_{r,c}⟩ = H_right[c,c]`` per sector block."""
function _diagonal_one_sided_aa!(aa::Vector{Float64}, heff::QCCenter, template::TensorMap)
	offs = _flat_sector_offsets(template)
	for (c, tpl_blk) in blocks(template)
		idx_base = offs[c]
		nrows, ncols = size(tpl_blk)
		if haskey(blocks(heff.Hleft), c)
			Hl = blocks(heff.Hleft)[c]
			@boundscheck size(Hl, 1) >= nrows
			@inbounds for c_loc in 1:ncols, r_loc in 1:nrows
				gi = idx_base + (c_loc - 1) * nrows + r_loc - 1
				aa[gi] += real(Hl[r_loc, r_loc])
			end
		end
		if haskey(blocks(heff.Hright), c)
			Hr = blocks(heff.Hright)[c]
			@boundscheck size(Hr, 2) >= ncols
			@inbounds for c_loc in 1:ncols, r_loc in 1:nrows
				gi = idx_base + (c_loc - 1) * nrows + r_loc - 1
				aa[gi] += real(Hr[c_loc, c_loc])
			end
		end
	end
	return aa
end

function _resolve_fusion_key(dict, f)
	haskey(dict, f) && return f
	for g in keys(dict)
		g == f && return g
	end
	error("fusion key $f not found in bond block layout")
end

function _bond_block_flat_info(template::TensorMap, f1r, offs::Dict{Any, Int})
	c = f1r.coupled
	haskey(offs, c) || error("sector $c missing from bond flat offsets")
	haskey(template.rowr, c) || error("sector $c missing from template.rowr")
	haskey(template.colr, c) || error("sector $c missing from template.colr")
	f_row = _resolve_fusion_key(template.rowr[c], f1r)
	f_col = _resolve_fusion_key(template.colr[c], f1r)
	return (
		offs[c],
		blocks(template)[c],
		template.rowr[c][f_row],
		template.colr[c][f_col],
	)
end

"""``y += ml * x * mr``, ``x = |e_{r_loc,c_loc}⟩``; accumulate ``y_{r_loc,c_loc} = ml[r,r] mr[c,c]`` into flat ``aa``."""
function _accumulate_twoside_block_diag!(
	aa::Vector{Float64}, idx_base::Int, blk::AbstractMatrix, rowr, colr,
	ml::AbstractMatrix, mr::AbstractMatrix, coeff::Real,
)
	nr, nc = length(rowr), length(colr)
	size(ml, 1) == nr || error("twoside ml row dim mismatch")
	size(ml, 2) == nr || error("twoside ml col dim mismatch")
	size(mr, 1) == nc || error("twoside mr row dim mismatch")
	size(mr, 2) == nc || error("twoside mr col dim mismatch")
	nblk = size(blk, 1)
	@inbounds for c_loc in 1:nc, r_loc in 1:nr
		r = rowr[r_loc]
		c = colr[c_loc]
		gi = idx_base + (c - 1) * nblk + r - 1
		aa[gi] += coeff * real(ml[r_loc, r_loc] * mr[c_loc, c_loc])
	end
	return aa
end

function _diagonal_cross_term_path!(
	aa::Vector{Float64}, template::TensorMap, left, right, f1l, f1r, coeff, add_adjoint, offs,
)
	ml = StridedView(dropdims(left[f1l, f1r], dims=2))
	f2r = FusionTree((f1l.uncoupled[1],), f1l.uncoupled[1], (false,))
	f2r == f1r || return nothing
	f2l = FusionTree((f1r.uncoupled[1], conj(f1l.uncoupled[2])), f2r.coupled, (false, false))
	rj = get(right, (f2l, f2r), nothing)
	isnothing(rj) && return nothing
	mr = StridedView(dropdims(rj, dims=2))
	xj = template[f1r, f1r]
	nr, nc = size(xj, 1), size(xj, 2)
	(size(ml, 1) == nr && size(ml, 2) == nr && size(mr, 1) == nc && size(mr, 2) == nc) || return nothing
	idx_base, blk, rowr, colr = _bond_block_flat_info(template, f1r, offs)
	(length(rowr) == nr && length(colr) == nc) || error("bond fusion block size mismatch")
	_accumulate_twoside_block_diag!(aa, idx_base, blk, rowr, colr, ml, mr, coeff)
	if add_adjoint
		_accumulate_twoside_block_diag!(aa, idx_base, blk, rowr, colr, ml', mr', coeff)
	end
	return nothing
end

"""Cross-term diagonal via fusion-block formula (same flat layout as `_vec_coeffs`)."""
function _diagonal_cross_terms_aa!(aa::Vector{Float64}, heff::QCCenter, template::TensorMap)
	offs = _flat_sector_offsets(template)
	for t in heff.terms
		for (f1l, f1r) in fusiontrees(t.left)
			_diagonal_cross_term_path!(
				aa, template, t.left, t.right, f1l, f1r, t.coeff, t.add_adjoint, offs,
			)
		end
	end
	return aa
end

"""Flat diagonal ``aa_i = ⟨e_i | H_eff | e_i⟩`` (block formulas for Hleft/Hright/cross terms)."""
function qc_diagonal_aa!(heff::QCCenter, template::TensorMap)
	n = dim(template)
	aa = zeros(Float64, n)
	_diagonal_one_sided_aa!(aa, heff, template)
	_diagonal_cross_terms_aa!(aa, heff, template)
	heff.diag_aa = aa
	return aa
end

function qc_diagonal_aa(heff::QCCenter, template::TensorMap)
	if heff.diag_aa !== nothing && length(heff.diag_aa) == dim(template)
		return heff.diag_aa
	end
	aa = qc_diagonal_aa!(heff, template)
	heff.diag_aa = aa
	return aa
end

function TK.mul!(y, m::QCCenter, x)
	t = active_dmrg_timing()
	if t !== nothing
		t.n_matvec += 1
	end
	mul!(y, m.Hleft, x, true, false)
	mul!(y, x, m.Hright, true, true)
	if !_use_parallel_terms(m)
		return _mul_twoside_terms_serial!(y, m, x)
	end
	return _mul_twoside_terms_parallel!(y, m, x)
end

(m::QCCenter)(x) = mul!(similar(x), m, x)

function _collect_twoside_paths(left::MPSTensor, right::MPSTensor; add_adjoint::Bool)
	paths = TwosideFusionPath[]
	for (f1l, f1r) in fusiontrees(left)
		f2r = FusionTree((f1l.uncoupled[1],), f1l.uncoupled[1], (false,))
		f2l = FusionTree((f1r.uncoupled[1], conj(f1l.uncoupled[2])), f2r.coupled, (false, false))
		isnothing(get(right, (f2l, f2r), nothing)) && continue
		push!(paths, TwosideFusionPath(f1l, f1r, false))
		add_adjoint && push!(paths, TwosideFusionPath(f1l, f1r, true))
	end
	return paths
end

function _apply_twoside_path!(
	y::MPSBondTensor, x::MPSBondTensor, left::MPSTensor, right::MPSTensor,
	path::TwosideFusionPath, workspace::Vector, α::Number,
)
	f1l, f1r = path.f1l, path.f1r
	ml = StridedView(dropdims(left[f1l, f1r], dims=2))
	f2r = FusionTree((f1l.uncoupled[1],), f1l.uncoupled[1], (false,))
	f2l = FusionTree((f1r.uncoupled[1], conj(f1l.uncoupled[2])), f2r.coupled, (false, false))
	rj = get(right, (f2l, f2r), nothing)
	isnothing(rj) && return y
	mr = StridedView(dropdims(rj, dims=2))
	if path.adjoint
		xj = x[f2r, f2r]
		yj = y[f1r, f1r]
		mul_twosides!(yj, ml', xj, mr', α, true, workspace)
	else
		xj = x[f1r, f1r]
		yj = y[f2r, f2r]
		mul_twosides!(yj, ml, xj, mr, α, true, workspace)
	end
	return y
end

const _fusion_parallel_ys = Ref{Union{Nothing, Vector{MPSBondTensor}}}(nothing)

function _ensure_fusion_thread_ys!(y::MPSBondTensor)
	cache = _fusion_parallel_ys[]
	nt = Threads.nthreads()
	if cache === nothing || length(cache) != nt
		cache = Vector{MPSBondTensor}(undef, nt)
		_fusion_parallel_ys[] = cache
	end
	for tid in 1:nt
		if !isassigned(cache, tid) || typeof(cache[tid]) != typeof(y) ||
				space(cache[tid]) != space(y)
			cache[tid] = similar(y)
		end
	end
	return cache
end

function _use_parallel_fusion(paths::AbstractVector{TwosideFusionPath})
	nt = Threads.nthreads()
	return nt > 1 && length(paths) >= nt * MIN_FUSION_PATHS_PER_THREAD
end

function _apply_twoside_paths_parallel!(
	y::MPSBondTensor, x::MPSBondTensor, left::MPSTensor, right::MPSTensor,
	paths::Vector{TwosideFusionPath}, workspace::Vector, α::Number,
)
	ys = _ensure_fusion_thread_ys!(y)
	nt = length(ys)
	for tid in 1:nt
		fill!(ys[tid], 0)
	end
	nexti = Threads.Atomic{Int}(1)
	tasks = Vector{Task}(undef, nt)
	for tid in 1:nt
		tasks[tid] = Threads.@spawn begin
			yt = ys[tid]
			ws = similar(workspace)
			while true
				i = Threads.atomic_add!(nexti, 1)
				i > length(paths) && break
				_apply_twoside_path!(yt, x, left, right, paths[i], ws, α)
			end
		end
	end
	wait.(tasks)
	for yt in ys
		axpy!(true, yt, y)
	end
	return y
end

function _apply_twoside_paths_serial!(
	y::MPSBondTensor, x::MPSBondTensor, left::MPSTensor, right::MPSTensor,
	paths::Vector{TwosideFusionPath}, workspace::Vector, α::Number,
)
	for path in paths
		_apply_twoside_path!(y, x, left, right, path, workspace, α)
	end
	return y
end

function apply_twosides!(
	y::MPSBondTensor, x::MPSBondTensor, left::MPSTensor, right::MPSTensor,
	workspace::Vector, α::Number=1;
	add_adjoint::Bool=false, parallel_fusion::Bool=true,
)
	paths = _collect_twoside_paths(left, right; add_adjoint=add_adjoint)
	if parallel_fusion && _use_parallel_fusion(paths)
		return _apply_twoside_paths_parallel!(y, x, left, right, paths, workspace, α)
	end
	return _apply_twoside_paths_serial!(y, x, left, right, paths, workspace, α)
end

function filter_pair_vector(a::Vector{A}, b::Vector{A}) where A
	@assert size(a) == size(b)
	anew = A[]
	bnew = A[]
	for i in 1:length(a)
		if isassigned(a, i) && isassigned(b, i)
			push!(anew, a[i])
			push!(bnew, b[i])
		end
	end
	return anew, bnew
end
