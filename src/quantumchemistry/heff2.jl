# Minimum independent terms per Julia thread before using @spawn parallel path.
const MIN_TERMS_PER_THREAD = 2

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

function TK.mul!(y, m::QCCenter, x)
	mul!(y, m.Hleft, x, true, false)
	mul!(y, x, m.Hright, true, true)
	if !_use_parallel_terms(m)
		return _mul_twoside_terms_serial!(y, m, x)
	end
	return _mul_twoside_terms_parallel!(y, m, x)
end

(m::QCCenter)(x) = mul!(similar(x), m, x)

function apply_twosides!(y::MPSBondTensor, x::MPSBondTensor, left::MPSTensor, right::MPSTensor, workspace::Vector, α::Number=1; add_adjoint::Bool)
	for (f1l, f1r) in fusiontrees(left)
		ml = StridedView(dropdims(left[f1l, f1r], dims=2))
		f2r = FusionTree((f1l.uncoupled[1],), f1l.uncoupled[1], (false,))
		f2l = FusionTree((f1r.uncoupled[1], conj(f1l.uncoupled[2])), f2r.coupled, (false, false))
		rj = get(right, (f2l, f2r), nothing)
		if !isnothing(rj)
			mr = StridedView(dropdims(rj, dims=2))
			xj = x[f1r, f1r]
			yj = y[f2r, f2r]
			mul_twosides!(yj, ml, xj, mr, α, true, workspace)
			if add_adjoint
				xj = x[f2r, f2r]
				yj = y[f1r, f1r]
				mul_twosides!(yj, ml', xj, mr', α, true, workspace)
			end
		end
	end
	return y
end

# to be optimized
function apply_twosides2!(y::MPSBondTensor, x::MPSBondTensor, left::MPSTensor, right::MPSTensor, workspace::Vector, α::Number=1; add_adjoint::Bool)
	@tensor y[1,5] += α * left[1,2,3] * x[3,4] * right[4,2,5]
	if add_adjoint
		@tensor y[3,5] += α * conj(left[1,2,3]) * x[1,4] * conj(right[5,2,4])
	end
	return y
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
