using LinearAlgebra: eigen, Hermitian
using SphericalTensors: scale!, axpy!, fill!, copy!, dot, norm, mul!, similar

"""
Davidson for the lowest eigenpair of a Hermitian operator.

**Terminology:**

- **Davidson** — subspace iteration: Rayleigh–Ritz, residual, subspace expansion, deflate.
- **Olsen preconditioner** — optional step inside Davidson (`olsen_precond=true`).
- **`NoPrecond`** — Davidson without Olsen (`olsen_precond=false`, default).
- **`DavidsonPrecond`** — alternate (diagonal divide only); not implemented here.

`conv_thrd` is the threshold on ``||H\\psi - E\\psi||`` (same as `QCDMRG2.toleig`).

Eigensolve vectors are **flat** bond tensors from `renormalizedoperator(x)` (same layout as `QCCenter.Hleft`/`Hright`).
"""
struct DavidsonSolver
	conv_thrd::Float64
	rel_conv_thrd::Float64
	max_iter::Int
	deflation_min_size::Int
	deflation_max_size::Int
end

DavidsonSolver(;
	conv_thrd::Real=1.0e-5,
	rel_conv_thrd::Real=0.0,
	max_iter::Int=5000,
	deflation_min_size::Int=2,
	deflation_max_size::Int=50,
) = DavidsonSolver(
	convert(Float64, conv_thrd),
	convert(Float64, rel_conv_thrd),
	max_iter,
	deflation_min_size,
	deflation_max_size,
)

function _tm_fill!(x::TensorMap, val::Real=0)
	fill!(x, val)
	return x
end

function _tm_normalize!(x::TensorMap)
	n = norm(x)
	(abs(n) > 1e-30) || error("Davidson: zero vector in normalize")
	scale!(x, 1 / n)
	return x
end

function _tm_axpy!(α, x::TensorMap, y::TensorMap)
	(α == 0) && return y
	axpy!(α, x, y)
	return y
end

function _tm_linear_combo!(dst::TensorMap, vecs, coeffs::AbstractVector, n::Int)
	_tm_fill!(dst, 0)
	for j in 1:n
		_tm_axpy!(coeffs[j], vecs[j], dst)
	end
	return dst
end

function _ritz_transform!(bs, sigmas, Vs::AbstractMatrix, m::Int, buf, bufσ)
	for j in 1:m
		_tm_linear_combo!(buf[j], bs, Vs[:, j], m)
		_tm_linear_combo!(bufσ[j], sigmas, Vs[:, j], m)
	end
	for j in 1:m
		copy!(bs[j], buf[j])
		copy!(sigmas[j], bufσ[j])
	end
	return nothing
end

function _vec_coeffs(x::TensorMap)
	v = Vector{real(scalartype(x))}(undef, dim(x))
	idx = 1
	for (_, blk) in blocks(x)
		for i in eachindex(blk)
			v[idx] = real(blk[i])
			idx += 1
		end
	end
	return v
end

function _set_coeffs!(x::TensorMap, v::AbstractVector)
	idx = 1
	for (_, blk) in blocks(x)
		for i in eachindex(blk)
			blk[i] = v[idx]
			idx += 1
		end
	end
	return x
end

"""Olsen preconditioner. Uses subspace vector ``c`` (Ritz ``bs[ick]`` after transform)."""
function olsen_precondition_flat!(q::TensorMap, c::TensorMap, λ::Real, aa::AbstractVector{<:Real})
	@assert length(aa) == dim(q)
	qf = _vec_coeffs(q)
	cf = _vec_coeffs(c)
	tf = similar(qf)
	for i in eachindex(qf)
		denom = λ - aa[i]
		tf[i] = abs(denom) > 1e-12 ? cf[i] / denom : cf[i]
	end
	ct = dot(cf, tf)
	for i in eachindex(qf)
		denom = λ - aa[i]
		qf[i] = abs(denom) > 1e-12 ? qf[i] / denom : qf[i]
	end
	if abs(ct) > 1e-30
		qf .-= (dot(cf, qf) / ct) .* tf
	end
	_set_coeffs!(q, qf)
	return q
end

"""Product diagonal for Olsen.

``aa_i = \\langle e_i | H_\\mathrm{left} + H_\\mathrm{right} | e_i \\rangle`` on the bond tensor
(one-sided applies only; cross terms are off-diagonal in this basis).
"""
function qc_diagonal_aa(heff::QCCenter, template::TensorMap)
	aa = Vector{Float64}(undef, dim(template))
	idx = 1
	tmp = similar(template)
	σ = similar(template)
	for (key, _) in blocks(template)
		n = length(blocks(template)[key])
		for i in 1:n
			_tm_fill!(tmp, 0)
			blocks(tmp)[key][i] = 1
			mul!(σ, heff.Hleft, tmp, true, false)
			mul!(σ, tmp, heff.Hright, true, true)
			aa[idx] = real(dot(tmp, σ))
			idx += 1
		end
	end
	return aa
end

function _orthog_subtract!(q::TensorMap, bs, m::Int)
	for j in 1:m
		nrm2 = real(dot(bs[j], bs[j]))
		if abs(nrm2) > 1e-30
			_tm_axpy!(-real(dot(bs[j], q)) / nrm2, bs[j], q)
		end
	end
	return q
end

function davidson_eigsolve(op, v0::TensorMap, solver::DavidsonSolver=DavidsonSolver();
		precond_aa=nothing)
	maxm = solver.deflation_max_size
	bs = [similar(v0) for _ in 1:maxm]
	sigmas = [similar(v0) for _ in 1:maxm]
	ritz_buf = [similar(v0) for _ in 1:maxm]
	ritz_bufσ = [similar(v0) for _ in 1:maxm]

	copy!(bs[1], v0)
	_tm_normalize!(bs[1])

	m = 1
	msig = 0
	n_mv = 0
	eigval = 0.0
	ick = 1
	res_norm = Inf
	converged = false

	q = similar(v0)

	for iter in 1:solver.max_iter
		while msig < m
			msig += 1
			mul!(sigmas[msig], op, bs[msig])
			n_mv += 1
		end

		Hproj = Matrix{Float64}(undef, m, m)
		for i in 1:m, j in 1:m
			Hproj[i, j] = real(dot(bs[i], sigmas[j]))
		end
		Hproj = (Hproj + Hproj') / 2
		λs, Vs = eigen(Hermitian(Hproj))
		_ritz_transform!(bs, sigmas, Vs, m, ritz_buf, ritz_bufσ)

		ick = argmin(λs)
		eigval = λs[ick]

		copy!(q, sigmas[ick])
		_tm_axpy!(-eigval, bs[ick], q)

		res_norm = norm(q)
		threshold = solver.conv_thrd + abs(eigval) * solver.rel_conv_thrd
		if res_norm < threshold
			converged = true
			break
		end

		if m >= solver.deflation_max_size
			m = msig = solver.deflation_min_size
			continue
		end

		if precond_aa !== nothing
			olsen_precondition_flat!(q, bs[ick], eigval, precond_aa)
		end
		_orthog_subtract!(q, bs, m)
		nq = norm(q)
		if nq < 1e-14
			m = msig = solver.deflation_min_size
			continue
		end
		scale!(q, 1 / nq)
		m += 1
		copy!(bs[m], q)
	end

	converged || error(
		"Davidson did not converge in $(solver.max_iter) iterations " *
		"(||r||=$(res_norm), tol=$(solver.conv_thrd), bond dim=$(dim(v0)))",
	)

	ψ_out = similar(v0)
	copy!(ψ_out, bs[ick])
	_tm_normalize!(ψ_out)
	return eigval, ψ_out, n_mv, res_norm
end
