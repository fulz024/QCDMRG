struct TwoSideTerm{A}
    left::Vector{A}
    right::Vector{A}
    coeff::Float64
    add_adjoint::Bool
end

struct QCCenter{A, B, T<:Number}
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
end

function QCCenter(left::QCSiteStorages, right::QCSiteStorages)
	# left = renormalizedstorage(left0)
	# right = renormalizedstorage(right0)
	adagTleft, adagTright = filter_pair_vector(left.adagT, right.adagT)
	Tdagaleft, Tdagaright = filter_pair_vector(left.Tdaga, right.Tdaga)
	Hleft, Hright = left.H, right.H
	n = max_blocksize(left.H)
	n = max(max_blocksize(right.H), n)
	workspace = zeros(scalartype(Hleft), n*n)
	res = QCCenter(left.H, right.H, adagTleft, adagTright, Tdagaleft, Tdagaright, left.PA, right.PA, left.BQ, right.BQ, workspace, [])
	append!(res.terms, _build_twoside_terms(res))
	return res
end

function _build_twoside_terms(m::QCCenter{A,B,T}) where {A,B,T}
    terms = TwoSideTerm{A}[]

    for (l,r) in zip(m.adagTleft, m.adagTright)
        push!(terms, TwoSideTerm(l,r,1.0,true))
    end

    for (l,r) in zip(m.Tdagaleft, m.Tdagaright)
        push!(terms, TwoSideTerm(l,r,1.0,true))
    end

    for r in 1:size(m.PAleft,1)
        for s in (r+1):size(m.PAleft,2)
            if isassigned(m.PAleft,r,s) && isassigned(m.PAright,r,s)
                push!(terms, TwoSideTerm(m.PAleft[r,s],m.PAright[r,s],1.0,true))
            end
        end
    end

    for r in 1:size(m.BQleft,1)

        if isassigned(m.BQleft,r,r) && isassigned(m.BQright,r,r)
            push!(terms, TwoSideTerm(m.BQleft[r,r],m.BQright[r,r],-1.0,false))
        end

        for s in (r+1):size(m.BQleft,2)
            if isassigned(m.BQleft,r,s) && isassigned(m.BQright,r,s)
                push!(terms, TwoSideTerm(m.BQleft[r,s],m.BQright[r,s],-1.0,true))
            end
        end
    end

    return terms
end

max_blocksize(x::AbstractTensorMap) = mapreduce(a->size(a[2], 1), max, blocks(x); init = 0)

function calc_galerkin(m::QCCenter, x::MPSBondTensor)
	out = m(x)
	try
		return norm(leftnull(x)' * out)
	catch
		return norm(out * rightnull(x)' )
	end
end

function TK.mul!(y, m::QCCenter, x)

    workspace = m.workspace
	workspaces = [similar(workspace) for _ in 1:Threads.nthreads()]

    mul!(y, m.Hleft, x, true, false)
    mul!(y, x, m.Hright, true, true)

    ys = [similar(y) for _ in 1:Threads.nthreads()]
    nt = Threads.nthreads()

    for i in 1:nt
        fill!(ys[i], 0)
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
					yt,
					x,
					t.left,
					t.right,
					wt,
					t.coeff,
					add_adjoint = t.add_adjoint
				)
			end
		end
	end
	wait.(tasks)

    for yt in ys
        y .+= yt
    end

    return y
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
