struct QCDMRG2 <: DMRGAlgorithm
	maxiter::Int
	tol::Float64
	maxitereig::Int
	toleig::Float64
	noise::Float64
	verbosity::Int
	trunc::TruncationDimCutoff
end

QCDMRG2(trunc::TruncationDimCutoff; maxiter::Int=100, tol::Real=1.0e-14, maxitereig::Int=10, toleig::Real=1.0e-10, noise::Real=0, verbosity::Int=1) = QCDMRG2(
	maxiter, convert(Float64, tol), maxitereig, convert(Float64, toleig), convert(Float64, noise), verbosity, trunc)
QCDMRG2(; trunc::TruncationDimCutoff=DMRG.DefaultTruncation, kwargs...) = QCDMRG2(trunc; kwargs...)

Base.similar(x::QCDMRG2; trunc::TruncationDimCutoff=x.trunc, maxiter::Int=x.maxiter, tol::Float64=x.tol, maxitereig::Int=x.maxitereig,
	toleig::Float64=x.toleig, verbosity::Int=x.verbosity) = QCDMRG2(trunc=trunc, maxiter=maxiter, tol=tol, maxitereig=maxitereig,
	toleig=toleig, verbosity=verbosity)

function Base.getproperty(x::QCDMRG2, s::Symbol)
	if s == :D
		return x.trunc.D
	elseif s == :ϵ
		return x.trunc.ϵ
	else
		getfield(x, s)
	end
end

function _timed!(f, t::DMRGTiming, field::Symbol)
	Δ = @elapsed ret = f()
	setproperty!(t, field, getproperty(t, field) + Δ)
	return ret
end

function _build_heff!(env::QCDMRGCache, bond::Int)
	mpsA, mpsB = env.mps[bond], env.mps[bond + 1]
	Sleft = renormalizestorageleft(env, bond, space_l(mpsA))
	Sright = renormalizestorageright(env, bond + 1, space_r(mpsB)')
	Opleft = renormalizedstorage(Sleft)
	Opright = renormalizedstorage(Sright)
	heff = QCCenter(Opleft, Opright)
	@tensor x[1, 2; 4, 5] := mpsA[1, 2, 3] * mpsB[3, 4, 5]
	return heff, x, Opleft, Opright, mpsA, mpsB
end

function _optimize_bond_left!(env::QCDMRGCache, bond::Int, alg::QCDMRG2)
	t = active_dmrg_timing()
	heff, x, Opleft, _, _, _ = if t === nothing
		_build_heff!(env, bond)
	else
		_timed!(() -> _build_heff!(env, bond), t, :teff)
	end

	eigenvalues_0, eigenvecs_0 = if t === nothing
		eigsolve(heff, renormalizedoperator(x), 1, :SR, Lanczos(; maxiter=100, tol=alg.toleig, eager=true))
	else
		_timed!(() -> eigsolve(heff, renormalizedoperator(x), 1, :SR, Lanczos(; maxiter=100, tol=alg.toleig, eager=true)), t, :teig)
	end
	eigenvalue_0, eigenvec_0 = eigenvalues_0[1], eigenvecs_0[1]
	eigenvec = TensorMap(blocks(eigenvec_0), codomain(x), domain(x))

	if alg.noise > 0
		if t === nothing
			noise_vec = TensorMap(randn, scalartype(eigenvec), space(eigenvec))
			axpy!(alg.noise, noise_vec, eigenvec)
		else
			_timed!(() -> begin
				noise_vec = TensorMap(randn, scalartype(eigenvec), space(eigenvec))
				axpy!(alg.noise, noise_vec, eigenvec)
			end, t, :teig)
		end
	end

	u, s, v, err = if t === nothing
		tsvd!(eigenvec, trunc=alg.trunc)
	else
		_timed!(() -> tsvd!(eigenvec, trunc=alg.trunc), t, :tsvd)
	end

	delta, eigenvalue, v2 = if t === nothing
		normalize!(s)
		v2 = s * v
		x′ = u * v2
		err_1 = dot(x′, x)
		δ = abs(1 - abs(err_1))
		x2′ = renormalizedoperator(x′)
		E = dot(x2′, heff(x2′))
		(δ, E, v2)
	else
		_timed!(() -> begin
			normalize!(s)
			v2 = s * v
			x′ = u * v2
			err_1 = dot(x′, x)
			δ = abs(1 - abs(err_1))
			x2′ = renormalizedoperator(x′)
			E = dot(x2′, heff(x2′))
			return (δ, E, v2)
		end, t, :tsplt)
	end

	if t === nothing
		env.mps[bond] = u
		env.mps[bond + 1] = permute(v2, (1, 2), (3,))
		Snew = updatestoragerenormalizeleft(Opleft, renormalizedoperator(env.mps[bond]))
		setstorage!(env, bond, Snew)
	else
		_timed!(() -> begin
			env.mps[bond] = u
			env.mps[bond + 1] = permute(v2, (1, 2), (3,))
			Snew = updatestoragerenormalizeleft(Opleft, renormalizedoperator(env.mps[bond]))
			setstorage!(env, bond, Snew)
		end, t, :tmve)
		t.nbonds += 1
	end

	(alg.verbosity > 2) && println("E₀=$(eigenvalue_0), E=$eigenvalue, δ=$(round(delta, digits=12)), χ=$(dim(space(s, 2))) after optimizing bond $bond")
	return eigenvalue, delta
end

function _optimize_bond_right!(env::QCDMRGCache, bond::Int, alg::QCDMRG2)
	t = active_dmrg_timing()
	heff, x, _, Opright, _, mpsB = if t === nothing
		_build_heff!(env, bond)
	else
		_timed!(() -> _build_heff!(env, bond), t, :teff)
	end

	eigenvalues_0, eigenvecs_0 = if t === nothing
		eigsolve(heff, renormalizedoperator(x), 1, :SR, Lanczos(; maxiter=100, tol=alg.toleig, eager=true))
	else
		_timed!(() -> eigsolve(heff, renormalizedoperator(x), 1, :SR, Lanczos(; maxiter=100, tol=alg.toleig, eager=true)), t, :teig)
	end
	eigenvalue_0, eigenvec_0 = eigenvalues_0[1], eigenvecs_0[1]
	eigenvec = TensorMap(blocks(eigenvec_0), codomain(x), domain(x))

	if alg.noise > 0
		if t === nothing
			noise_vec = TensorMap(randn, scalartype(eigenvec), space(eigenvec))
			axpy!(alg.noise, noise_vec, eigenvec)
		else
			_timed!(() -> begin
				noise_vec = TensorMap(randn, scalartype(eigenvec), space(eigenvec))
				axpy!(alg.noise, noise_vec, eigenvec)
			end, t, :teig)
		end
	end

	u, s, v, err = if t === nothing
		tsvd!(eigenvec, trunc=alg.trunc)
	else
		_timed!(() -> tsvd!(eigenvec, trunc=alg.trunc), t, :tsvd)
	end

	delta, eigenvalue, u2, mpsB = if t === nothing
		normalize!(s)
		u2 = u * s
		x′ = u2 * v
		err_1 = dot(x′, x)
		δ = abs(1 - abs(err_1))
		x2′ = renormalizedoperator(x′)
		E = dot(x2′, heff(x2′))
		(δ, E, u2, v)
	else
		_timed!(() -> begin
			normalize!(s)
			u2 = u * s
			x′ = u2 * v
			err_1 = dot(x′, x)
			δ = abs(1 - abs(err_1))
			x2′ = renormalizedoperator(x′)
			E = dot(x2′, heff(x2′))
			return (δ, E, u2, v)
		end, t, :tsplt)
	end

	if t === nothing
		env.mps[bond] = u2
		env.mps[bond + 1] = permute(mpsB, (1, 2), (3,))
		env.mps.s[bond + 1] = s
		Snew = updatestoragerenormalizeright(Opright, renormalizedoperator(mpsB))
		setstorage!(env, bond + 1, Snew)
	else
		_timed!(() -> begin
			env.mps[bond] = u2
			env.mps[bond + 1] = permute(mpsB, (1, 2), (3,))
			env.mps.s[bond + 1] = s
			Snew = updatestoragerenormalizeright(Opright, renormalizedoperator(mpsB))
			setstorage!(env, bond + 1, Snew)
		end, t, :tmve)
		t.nbonds += 1
	end
	(alg.verbosity > 2) && println("E₀=$(eigenvalue_0), E=$eigenvalue, δ=$(round(delta, digits=12)), χ=$(dim(space(s, 2))) after optimizing bond $bond")
	return eigenvalue, delta
end

function _run_half_sweep_timing!(alg::QCDMRG2, direction::String, f)
	reset_dmrg_timing!()
	energies, delta = f()
	t = finish_dmrg_timing!()
	if alg.verbosity >= 1
		print_dmrg_timing(t; direction=direction)
	end
	return energies, delta, t
end

function DMRG.leftsweep!(env::QCDMRGCache, alg::QCDMRG2)
	return _run_half_sweep_timing!(alg, "forward") do
		energies = Float64[]
		delta = 0.0
		for bond in 1:length(env) - 2
			(alg.verbosity > 3) && println("sweeping from left to right at bond: $bond")
			eigenvalue, δ = _optimize_bond_left!(env, bond, alg)
			delta = max(delta, δ)
			push!(energies, eigenvalue)
		end
		return energies, delta
	end
end

function DMRG.rightsweep!(env::QCDMRGCache, alg::QCDMRG2)
	return _run_half_sweep_timing!(alg, "backward") do
		energies = Float64[]
		delta = 0.0
		for bond in length(env) - 1:-1:1
			(alg.verbosity > 3) && println("sweeping from right to left at bond: $bond")
			eigenvalue, δ = _optimize_bond_right!(env, bond, alg)
			delta = max(delta, δ)
			push!(energies, eigenvalue)
		end
		return energies, delta
	end
end

function DMRG.sweep!(m::QCDMRGCache, alg::QCDMRG2)
	_, delta1, t_fwd = leftsweep!(m, alg)
	Energies2, delta2, t_bwd = rightsweep!(m, alg)
	delta = max(delta1, delta2)

	if alg.verbosity > 1
		println("E=$(Energies2[end]), δ=$(round(delta, digits=12)) after a full sweep")
		println()
	end
	return Energies2[end], delta, DMRGSweepTiming(t_fwd, t_bwd)
end

function _eigsolve(h, init, maxiter, tol)
	if dim(init) >= 20
		eigenvalue_0, eigenvec_0, info = DMRG.simple_lanczos_solver(h, init, "SR", maxiter, tol, verbosity=0)
	else
		init = TensorMap(randn, scalartype(init), space(init))
		eigenvalues, eigenvecs, infos = eigsolve(h, init, 1, :SR, Lanczos())
		eigenvalue_0 = eigenvalues[1]
		eigenvec_0 = eigenvecs[1]
	end
	return eigenvalue_0, eigenvec_0
end
