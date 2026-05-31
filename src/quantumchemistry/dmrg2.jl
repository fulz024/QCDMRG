struct QCDMRG2 <: DMRGAlgorithm
	maxiter::Int
	tol::Float64
	maxitereig::Int
	toleig::Float64
	noise::Float64
	verbosity::Int
	trunc::TruncationDimCutoff
	eigsolver::Symbol
	davidson_max_subspace::Int
	davidson::Union{Nothing, DavidsonSolver}
end

# `toleig`: bond eigsolve convergence on ||Hψ - Eψ|| (KrylovKit `Lanczos.tol` / Davidson `conv_thrd`).
# `davidson`: optional `DavidsonSolver` (precond, subspace size, …); default uses `DavidsonSolver(deflation_max_size=davidson_max_subspace)`.
function QCDMRG2(trunc::TruncationDimCutoff; maxiter::Int=100, tol::Real=1.0e-14, maxitereig::Int=10, toleig::Real=1.0e-5, noise::Real=1.0e-10, verbosity::Int=1,
	eigsolver::Symbol=:davidson, davidson_max_subspace::Int=50, davidson::Union{Nothing, DavidsonSolver}=nothing)
	eigsolver in (:davidson, :lanczos) ||
		error("eigsolver must be :davidson or :lanczos; got $(eigsolver)")
	return QCDMRG2(
		maxiter, convert(Float64, tol), maxitereig, convert(Float64, toleig), convert(Float64, noise), verbosity, trunc,
		eigsolver, davidson_max_subspace, davidson,
	)
end
QCDMRG2(; trunc::TruncationDimCutoff=DMRG.DefaultTruncation, kwargs...) = QCDMRG2(trunc; kwargs...)

Base.similar(x::QCDMRG2; trunc::TruncationDimCutoff=x.trunc, maxiter::Int=x.maxiter, tol::Float64=x.tol, maxitereig::Int=x.maxitereig,
	toleig::Float64=x.toleig, verbosity::Int=x.verbosity, eigsolver::Symbol=x.eigsolver,
	davidson_max_subspace::Int=x.davidson_max_subspace, davidson::Union{Nothing, DavidsonSolver}=x.davidson) = QCDMRG2(
	trunc=trunc, maxiter=maxiter, tol=tol, maxitereig=maxitereig, toleig=toleig, verbosity=verbosity,
	eigsolver=eigsolver, davidson_max_subspace=davidson_max_subspace, davidson=davidson)

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
	if field === :tmve_heavy || field === :tmve_light
		t.tmve += Δ
	end
	return ret
end

"""Shift QC environments to the current bond."""
function _renormalize_bond_storages!(env::QCDMRGCache, bond::Int)
	mpsA, mpsB = env.mps[bond], env.mps[bond + 1]
	Sleft = renormalizestorageleft(env, bond, space_l(mpsA))
	Sright = renormalizestorageright(env, bond + 1, space_r(mpsB)')
	return Sleft, Sright, mpsA, mpsB
end

"""Assemble `QCCenter` for Lanczos."""
function _assemble_heff!(Sleft, Sright, mpsA, mpsB)
	Opleft = renormalizedstorage(Sleft)
	Opright = renormalizedstorage(Sright)
	heff = QCCenter(Opleft, Opright)
	@tensor x[1, 2; 4, 5] := mpsA[1, 2, 3] * mpsB[3, 4, 5]
	return heff, x, Opleft, Opright
end

function _prepare_bond_heff!(env::QCDMRGCache, bond::Int, t; compute_aa::Bool=false)
	Sleft, Sright, mpsA, mpsB = if t === nothing
		_renormalize_bond_storages!(env, bond)
	else
		_timed!(() -> _renormalize_bond_storages!(env, bond), t, :tmve_heavy)
	end
	heff, x, Opleft, Opright = if t === nothing
		heff, x, Opleft, Opright = _assemble_heff!(Sleft, Sright, mpsA, mpsB)
		if compute_aa
			qc_diagonal_aa!(heff, renormalizedoperator(x))
		end
		heff, x, Opleft, Opright
	else
		_timed!(() -> begin
			heff, x, Opleft, Opright = _assemble_heff!(Sleft, Sright, mpsA, mpsB)
			if compute_aa
				qc_diagonal_aa!(heff, renormalizedoperator(x))
			end
			return heff, x, Opleft, Opright
		end, t, :teff)
	end
	return heff, x, Opleft, Opright, mpsA, mpsB
end

function _needs_precond_aa(alg::QCDMRG2)
	alg.eigsolver === :davidson || return false
	cfg = something(alg.davidson, DavidsonSolver(deflation_max_size=alg.davidson_max_subspace))
	return cfg.precond !== :none
end

"""Initial guess on flat bond space (matches `QCCenter` `Hleft`/`Hright` layout)."""
function _eig_init(x)
	ψ0 = renormalizedoperator(x)
	normalize!(ψ0)
	return ψ0
end

function _log_bond_eig!(rec::BondEigRecord, alg::QCDMRG2, t, direction::String)
	if t !== nothing
		push!(t.bond_records, rec)
	end
	if alg.verbosity >= 2
		print_bond_eig_record(rec; direction=direction, D=alg.D, tol=alg.toleig)
	end
	return rec
end

function _lanczos_solver(heff, x, alg::QCDMRG2, bond::Int, t, direction::String)
	ψ0 = _eig_init(x)
	lanczos = Lanczos(; maxiter=100, tol=alg.toleig, eager=true)
	n_mv0 = t === nothing ? 0 : t.n_matvec
	solve = () -> eigsolve(heff, ψ0, 1, :SR, lanczos)
	eigenvalues, eigenvecs, info = if t === nothing
		solve()
	else
		_timed!(solve, t, :teig)
	end
	nmv = t === nothing ? info.numops : t.n_matvec - n_mv0
	normres = isempty(info.normres) ? NaN : info.normres[1]
	rec = BondEigRecord(bond, eigenvalues[1], normres, nmv, info.numiter)
	_log_bond_eig!(rec, alg, t, direction)
	return eigenvalues, eigenvecs, info
end

function _davidson_solver(heff::QCCenter, x, alg::QCDMRG2, bond::Int, t, direction::String)
	ψ0 = _eig_init(x)
	cfg = something(alg.davidson, DavidsonSolver(deflation_max_size=alg.davidson_max_subspace))
	solver = DavidsonSolver(;
		conv_thrd=alg.toleig,
		rel_conv_thrd=cfg.rel_conv_thrd,
		max_iter=cfg.max_iter,
		deflation_min_size=cfg.deflation_min_size,
		deflation_max_size=cfg.deflation_max_size,
		precond=cfg.precond,
	)
	solve = () -> davidson_eigsolve(heff, ψ0, solver)
	eigval, ψ_out, nmv, normres = if t === nothing
		solve()
	else
		_timed!(solve, t, :teig)
	end
	rec = BondEigRecord(bond, eigval, normres, nmv, 0)
	_log_bond_eig!(rec, alg, t, direction)
	eigenvalues = [eigval]
	eigenvecs = [ψ_out]
	return eigenvalues, eigenvecs, (normres=normres, numops=nmv)
end

function _bond_eigsolve(heff, x, alg::QCDMRG2, bond::Int, t, direction::String)
	if alg.eigsolver === :davidson
		return _davidson_solver(heff, x, alg, bond, t, direction)
	elseif alg.eigsolver === :lanczos
		return _lanczos_solver(heff, x, alg, bond, t, direction)
	else
		error("unknown eigsolver $(alg.eigsolver); use :davidson or :lanczos")
	end
end

function _optimize_bond_left!(env::QCDMRGCache, bond::Int, alg::QCDMRG2)
	t = active_dmrg_timing()
	heff, x, Opleft, _, _, _ = _prepare_bond_heff!(env, bond, t; compute_aa=_needs_precond_aa(alg))

	eigenvalues_0, eigenvecs_0, _ = _bond_eigsolve(heff, x, alg, bond, t, "forward")
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
		end, t, :tmve_light)
		t.nbonds += 1
	end

	(alg.verbosity > 2) && println("E₀=$(eigenvalue_0), E=$eigenvalue, δ=$(round(delta, digits=12)), χ=$(dim(space(s, 2))) after optimizing bond $bond")
	return eigenvalue, delta
end

function _optimize_bond_right!(env::QCDMRGCache, bond::Int, alg::QCDMRG2)
	t = active_dmrg_timing()
	heff, x, _, Opright, _, mpsB = _prepare_bond_heff!(env, bond, t; compute_aa=_needs_precond_aa(alg))

	eigenvalues_0, eigenvecs_0, _ = _bond_eigsolve(heff, x, alg, bond, t, "backward")
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
		end, t, :tmve_light)
		t.nbonds += 1
	end
	(alg.verbosity > 2) && println("E₀=$(eigenvalue_0), E=$eigenvalue, δ=$(round(delta, digits=12)), χ=$(dim(space(s, 2))) after optimizing bond $bond")
	return eigenvalue, delta
end

# `f do ... end` passes the closure as the first argument.
function _run_half_sweep_timing!(f, alg::QCDMRG2, direction::String)
	configure_threading!(blas_threads=active_blas_threads_for_julia())
	reset_renorm_scratch_pools!()
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
