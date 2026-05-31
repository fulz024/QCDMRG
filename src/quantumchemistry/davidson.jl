using LinearAlgebra: eigen, Hermitian
using SphericalTensors: scale!, axpy!, fill!, copy!, dot, norm, mul!, similar

"""
Davidson for the lowest eigenpair of a Hermitian operator.

**Terminology:**

- **Davidson** — subspace iteration: Rayleigh–Ritz, residual, subspace expansion, deflate.
- **`NoPrecond`** — `precond=:none`.
- **`DavidsonPrecond`** — `precond=:davidson`: ``q_i ← q_i / (λ - aa_i)`` on ``diag(H_\\mathrm{eff})`` (``aa`` built in `Teff` via `qc_diagonal_aa!`).
- **Olsen** — `precond=:olsen` (default): same ``aa`` plus projection (block2 `Normal`).

`conv_thrd` is the threshold on ``||H\\psi - E\\psi||`` (same as `QCDMRG2.toleig`).

Eigensolve vectors are **flat** bond tensors from `renormalizedoperator(x)` (same layout as `QCCenter.Hleft`/`Hright`).
"""
struct DavidsonSolver
	conv_thrd::Float64
	rel_conv_thrd::Float64
	max_iter::Int
	deflation_min_size::Int
	deflation_max_size::Int
	precond::Symbol  # :none | :davidson | :olsen
end

const _DAVIDSON_PRECONDS = (:none, :davidson, :olsen)

DavidsonSolver(;
	conv_thrd::Real=1.0e-5,
	rel_conv_thrd::Real=0.0,
	max_iter::Int=5000,
	deflation_min_size::Int=2,
	deflation_max_size::Int=50,
	precond::Symbol=:olsen,
) = begin
	precond in _DAVIDSON_PRECONDS ||
		error("precond must be one of $(_DAVIDSON_PRECONDS); got $(precond)")
	DavidsonSolver(
		convert(Float64, conv_thrd),
		convert(Float64, rel_conv_thrd),
		max_iter,
		deflation_min_size,
		deflation_max_size,
		precond,
	)
end

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

"""Diagonal preconditioner (block2 `DavidsonPrecond`): ``q_i ← q_i / (λ - aa_i)`` only."""
function davidson_precondition_flat!(q::TensorMap, λ::Real, aa::AbstractVector{<:Real})
	@assert length(aa) == dim(q)
	qf = _vec_coeffs(q)
	for i in eachindex(qf)
		denom = λ - aa[i]
		qf[i] = abs(denom) > 1e-12 ? qf[i] / denom : qf[i]
	end
	_set_coeffs!(q, qf)
	return q
end

"""Olsen preconditioner (block2 `Normal`). Uses subspace vector ``c`` (Ritz ``bs[ick]`` after transform)."""
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

"""MGS vs current subspace after precond (block2 `davidson`: `-⟨b_j|q⟩`, Ritz basis orthonormal)."""
function _orthog_subtract!(q::TensorMap, bs, m::Int)
	for j in 1:m
		_tm_axpy!(-real(dot(bs[j], q)), bs[j], q)
	end
	return q
end

function davidson_eigsolve(op, v0::TensorMap, solver::DavidsonSolver=DavidsonSolver())
	precond = solver.precond
	precond_aa = if precond === :none
		nothing
	else
		op isa QCCenter || error("precond=$(precond) requires op to be QCCenter")
		aa = op.diag_aa
		(aa !== nothing && length(aa) == dim(v0)) ||
			error("precond=$(precond) requires heff.diag_aa from qc_diagonal_aa! during bond Heff assembly")
		aa
	end
	maxm = solver.deflation_max_size
	bs = [similar(v0) for _ in 1:maxm]
	sigmas = [similar(v0) for _ in 1:maxm]
	ritz_buf = [similar(v0) for _ in 1:maxm]
	ritz_bufσ = [similar(v0) for _ in 1:maxm]

	copy!(bs[1], v0)
	_tm_normalize!(bs[1])

	m = 1
	msig = 0
	ndav = 0
	eigval = 0.0
	ick = 1
	res_norm = Inf
	res_norm_sq = Inf
	converged = false

	q = similar(v0)

	for iter in 1:solver.max_iter
		while msig < m
			msig += 1
			mul!(sigmas[msig], op, bs[msig])
			ndav += 1
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

		res_norm_sq = norm(q)^2
		threshold_sq = solver.conv_thrd^2 +
			abs(eigval)^2 * solver.rel_conv_thrd^2
		if res_norm_sq < threshold_sq
			converged = true
			res_norm = sqrt(res_norm_sq)
			break
		end

		# block2 regular davidson: precond → deflate if full → orthog vs bs → normalize → add
		if precond === :davidson
			davidson_precondition_flat!(q, eigval, precond_aa)
		elseif precond === :olsen
			olsen_precondition_flat!(q, bs[ick], eigval, precond_aa)
		end
		if m >= solver.deflation_max_size
			m = msig = solver.deflation_min_size
		end
		_orthog_subtract!(q, bs, m)
		nq = norm(q)
		(nq > 1e-30) || continue
		scale!(q, 1 / nq)
		m += 1
		copy!(bs[m], q)
		res_norm = nq
	end

	if !converged
		res_norm = sqrt(res_norm_sq)
	end

	converged || error(
		"Davidson did not converge in $(solver.max_iter) iterations " *
		"(||r||=$(res_norm), tol=$(solver.conv_thrd), bond dim=$(dim(v0)))",
	)

	ψ_out = similar(v0)
	copy!(ψ_out, bs[ick])
	_tm_normalize!(ψ_out)
	return eigval, ψ_out, ndav, res_norm
end
