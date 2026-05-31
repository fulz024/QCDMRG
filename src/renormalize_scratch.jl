"""
Thread-local scratch buffers for `renormalizestorage*` / `updatestoragerenormalize*`
"""

"""Per-call cache for small physical-site operator TensorMaps."""
mutable struct TensorMapCache
	left::Dict{Any, Any}
	right::Dict{Any, Any}
	scope::Any
end

TensorMapCache() = TensorMapCache(Dict{Any, Any}(), Dict{Any, Any}(), nothing)

mutable struct RenormalizeScratch{T<:Number}
	workspace::Vector{T}
	workspace_cap::Int
	mat_pool::Dict{Tuple{Any,Int,Int}, Any}
	vec_pool::Dict{Tuple{Any,Int}, Any}
	rtensor_template_pool::Dict{Tuple{Any,Any,Any}, Any}
	tmcache::Any
end

RenormalizeScratch{T}() where {T<:Number} = RenormalizeScratch{T}(T[], 0, Dict(), Dict(), Dict(), TensorMapCache())

const _renorm_scratch = RenormalizeScratch{Float64}[RenormalizeScratch{Float64}() for _ in 1:Threads.maxthreadid()]

@inline function current_renorm_scratch()
	tid = Threads.threadid()
	while length(_renorm_scratch) < tid
		push!(_renorm_scratch, RenormalizeScratch{Float64}())
	end
	return _renorm_scratch[tid]
end

"""Grow-or-reuse numeric workspace for `mul_twosides!` / renormalize updates."""
function scratch_workspace!(scratch::RenormalizeScratch{T}, ::Type{T}, mpsj::AbstractTensorMap) where {T<:Number}
	n = compute_workspace(mpsj)
	if length(scratch.workspace) < n
		resize!(scratch.workspace, n)
		scratch.workspace_cap = n
	end
	return scratch.workspace
end

function scratch_workspace!(mpsj::AbstractTensorMap)
	T = scalartype(mpsj)
	return scratch_workspace!(current_renorm_scratch(), T, mpsj)
end

const MIN_RENORM_TASKS_FOR_THREADS = 4

"""Cached creation / annihilation operators for a two-orbital site `sc`."""
struct SiteOps
	adag::NTuple{2, Any}
	ann::NTuple{2, Any}
	sgn::Any
end

function SiteOps(sc)
	@assert length(sc) == 2
	return SiteOps(
		(sqC(sc, 1, true), sqC(sc, 2, true)),
		(sqC(sc, 1, false), sqC(sc, 2, false)),
		sgnC(sc),
	)
end

@inline site_adag(op::SiteOps, i::Int) = op.adag[i]
@inline site_ann(op::SiteOps, i::Int) = op.ann[i]

"""`op_pq = Σ_{p<q} h2e[p,q,orbr,orbs] * a_p^† a_q` using cached site operators."""
function h2e_pair_op_pq(sc, h2e, orbr, orbs, ops::SiteOps)
	op_pq = scratch_empty()
	for (idxp, orbp) in enumerate(sc)
		op_p = site_adag(ops, idxp)
		for (idxq, orbq) in enumerate(sc)
			if orbp < orbq
				op_pq += h2e[orbp, orbq, orbr, orbs] * op_p * site_adag(ops, idxq)
			end
		end
	end
	return op_pq
end

"""Fresh empty fermionic operator (avoid mutating shared `_empty`)."""
@inline scratch_empty() = empty_operator()

"""Reuse `Matrix{A}(undef, m, n)` when dimensions match."""
function scratch_matrix!(scratch::RenormalizeScratch, ::Type{A}, m::Int, n::Int) where {A}
	key = (A, m, n)
	if haskey(scratch.mat_pool, key)
		return scratch.mat_pool[key]::Matrix{A}
	end
	mat = Matrix{A}(undef, m, n)
	scratch.mat_pool[key] = mat
	return mat
end

scratch_matrix!(::Type{A}, m::Int, n::Int) where {A} =
	scratch_matrix!(current_renorm_scratch(), A, m, n)

"""Reuse `Vector{A}(undef, n)`."""
function scratch_vector!(scratch::RenormalizeScratch, ::Type{A}, n::Int) where {A}
	key = (A, n)
	if haskey(scratch.vec_pool, key)
		return scratch.vec_pool[key]::Vector{A}
	end
	vec = Vector{A}(undef, n)
	scratch.vec_pool[key] = vec
	return vec
end

scratch_vector!(::Type{A}, n::Int) where {A} = scratch_vector!(current_renorm_scratch(), A, n)

"""Create a zero `RATensor` while reusing cached block-structure templates."""
function scratch_rtensor!(
	scratch::RenormalizeScratch, ::Type{T},
	codom::ProductSpace{S,3}, dom::ProductSpace{S,2},
) where {T<:Number,S<:ElementarySpace}
	key = (T, codom, dom)
	template = get!(scratch.rtensor_template_pool, key) do
		RATensor(zeros, T, codom, dom)
	end
	t = similar(template, T)
	fill!(t.data, zero(T))
	return t
end

scratch_rtensor!(::Type{T}, codom::ProductSpace{S,3}, dom::ProductSpace{S,2}) where {T<:Number,S<:ElementarySpace} =
	scratch_rtensor!(current_renorm_scratch(), T, codom, dom)

function scratch_tensormap_cache!(scratch::RenormalizeScratch, scope)
	cache = scratch.tmcache::TensorMapCache
	if cache.scope != scope
		empty!(cache.left)
		empty!(cache.right)
		cache.scope = scope
	end
	return cache
end

scratch_tensormap_cache!(scope) = scratch_tensormap_cache!(current_renorm_scratch(), scope)

function cached_tensormap!(cache::TensorMapCache, key, op; side::Symbol)
	d = side === :L ? cache.left : cache.right
	return get!(d, key) do
		totensormap(op; side=side)
	end
end

"""Fresh operator cache for one storage cell (`@threads` must not share thread-local cache)."""
@inline storage_cell_tmcache() = TensorMapCache()

function reset_renorm_scratch_pools!()
	for s in _renorm_scratch
		empty!(s.mat_pool)
		empty!(s.vec_pool)
		empty!(s.rtensor_template_pool)
		empty!(s.tmcache.left)
		empty!(s.tmcache.right)
		s.tmcache.scope = nothing
	end
	return nothing
end

"""One `(idxr, idxs)` block of the left-storage PA update (thread-safe: unique matrix cell)."""
function _fill_one_PA_left!(
	PAnew, PAold, adagTold, id_left,
	idxr::Int, idxs::Int, orbr, orbs, sc, sl, h2e, ops::SiteOps,
	tmcache::Union{Nothing, TensorMapCache}=nothing,
)
	if isassigned(PAold, idxr + 2, idxs + 2)
		PAnew[idxr, idxs] = renormalizeleft(PAold[idxr + 2, idxs + 2], nothing)
	end
	op_pq = h2e_pair_op_pq(sc, h2e, orbr, orbs, ops)
	if !iszero(op_pq)
		if isassigned(PAnew, idxr, idxs)
			PAnew[idxr, idxs] = renormalizeleft!(PAnew[idxr, idxs], id_left, totensormap(op_pq, side=:L))
		else
			PAnew[idxr, idxs] = renormalizeleft(id_left, totensormap(op_pq, side=:L))
		end
	end
	for (idxp, orbp) in enumerate(sl)
		for (idxq, orbq) in enumerate(sc)
			coef = h2e[orbp, orbq, orbr, orbs]
			if !iszero(coef)
				op_q = site_adag(ops, idxq)
				op_q_t = tmcache === nothing ?
					totensormap(coef * op_q, side=:L) :
					coef * cached_tensormap!(tmcache, (:adag, idxq), op_q; side=:L)
				if isassigned(PAnew, idxr, idxs)
					PAnew[idxr, idxs] = renormalizeleft!(PAnew[idxr, idxs], adagTold[idxp], op_q_t)
				else
					PAnew[idxr, idxs] = renormalizeleft(adagTold[idxp], op_q_t)
				end
			end
		end
	end
	return nothing
end

function fill_PA_left!(PAnew, PAold, adagTold, id_left, sr, sc, sl, h2e)
	ops = SiteOps(sc)
	if Threads.nthreads() == 1
		tmcache = TensorMapCache()
		for (idxr, orbr) in enumerate(sr)
			for (idxs, orbs) in enumerate(sr)
				if orbr < orbs
					_fill_one_PA_left!(PAnew, PAold, adagTold, id_left, idxr, idxs, orbr, orbs, sc, sl, h2e, ops, tmcache)
				end
			end
		end
		return PAnew
	end
	pairs = NTuple{4,Any}[]
	for (idxr, orbr) in enumerate(sr)
		for (idxs, orbs) in enumerate(sr)
			if orbr < orbs
				push!(pairs, (idxr, idxs, orbr, orbs))
			end
		end
	end
	if length(pairs) >= MIN_RENORM_TASKS_FOR_THREADS
		Threads.@threads for (idxr, idxs, orbr, orbs) in pairs
			tmcache = storage_cell_tmcache()
			_fill_one_PA_left!(PAnew, PAold, adagTold, id_left, idxr, idxs, orbr, orbs, sc, sl, h2e, ops, tmcache)
		end
	else
		tmcache = TensorMapCache()
		for (idxr, idxs, orbr, orbs) in pairs
			_fill_one_PA_left!(PAnew, PAold, adagTold, id_left, idxr, idxs, orbr, orbs, sc, sl, h2e, ops, tmcache)
		end
	end
	return PAnew
end
