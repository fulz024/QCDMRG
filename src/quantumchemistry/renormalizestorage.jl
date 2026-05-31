function renormalizeHright(env::QCDMRGCache, site::Int, mpsj::MPSTensor=env.mps[site])
	if site == length(env)
		return renormalizeHright(env.ham, space_r(mpsj)')
	else
		return renormalizeHright(storage(env, site+1), env.ham, site, space_r(mpsj)')
	end
end

function updateHright(env::QCDMRGCache, site::Int, mpsj::MPSTensor=env.mps[site])
	hnew = renormalizeHright(env, site, mpsj)
	return updaterenormalizeright(hnew, mpsj, mpsj)
end

function renormalizeHright(ham::MolecularHamiltonian, spacer::ElementarySpace)
	hj = hlocal(ham, length(ham))
	id_right = isomorphism(storagetype(hj), spacer, spacer)
	hnew = renormalizeright(id_right, hj)
	_issymmetric(hnew) || throw(ArgumentError("h matrix is not symmetric"))
	return hnew
end
function renormalizeHright(storage_old::QCSiteStorages, ham::MolecularHamiltonian, site::Int, spacer::ElementarySpace)
	L = length(ham)
	(1 <= site < L) || throw(BoundsError())
	Hold, BQold, PAold, adagTold, Tdagaold = storage_old.H, storage_old.BQ, storage_old.PA, storage_old.adagT, storage_old.Tdaga
	h1e, h2e = ham.h1e, ham.h2e
	sl, sc, sr = get_splitting(L, site)
	nl, nr = get_nl_nr(L, site)

	id_right = isomorphism(storagetype(Hold), spacer, spacer)
	hnew = renormalizeright(id_right, hlocal(ham, site))

	@assert length(adagTold) == nl + 2
	@assert length(Tdagaold) == nr 
	@assert size(PAold, 1) == nr
	@assert size(BQold, 1) == nl + 2

	hnew = renormalizeright!(hnew, Hold, isomorphism(_u1u1_pspace, _u1u1_pspace))
	# eat aT
	for (idxp, orbp) in enumerate(sc)
		op_p =  sqC(sc, idxp, true) * sgnC(sc)
		if isassigned(adagTold, orbp)
			hnew = renormalizeright!(hnew, adagTold[orbp], totensormap(op_p, side=:L), add_adjoint=true)
		end
	end

	# eat Ta
	for (idxs, orbs) in enumerate(sr)
		op_pqr = scratch_empty()
		for (idxp, orbp) in enumerate(sc)
			op_p = sqC(sc, idxp, true)
			for (idxq, orbq) in enumerate(sc)
				op_q = sqC(sc, idxq, true)
				if orbp < orbq
					op_pq = op_p * op_q
					for (idxr, orbr) in enumerate(sc)
						op_r = sqC(sc, idxr, false)
						op_pqr += h2e[orbp, orbq, orbr, orbs] * op_pq * op_r * sgnC(sc)
					end
				end
			end
		end
		if !iszero(op_pqr) 
			hnew = renormalizeright!(hnew, Tdagaold[idxs], totensormap(op_pqr, side=:L), add_adjoint=true)
		end
	end
	# eat A
	for (idxr, orbr) in enumerate(sr)
		for (idxs, orbs) in enumerate(sr)
			if orbr < orbs
				op_pq = scratch_empty()
				for (idxp, orbp) in enumerate(sc)
					op_p = sqC(sc, idxp, true)
					for (idxq, orbq) in enumerate(sc)
						op_q = sqC(sc, idxq, true)
						coef = h2e[orbp, orbq, orbr, orbs]
						op_pq += coef * op_p * op_q
					end
				end
				if !iszero(op_pq)
					hnew = renormalizeright!(hnew, PAold[idxr, idxs], totensormap(op_pq, side=:L), add_adjoint=true)
				end
			end
		end
	end
	# eat B
	for (idxp, orbp) in enumerate(sc)
		op_p = sqC(sc, idxp, true)
		for (idxr, orbr) in enumerate(sc)
			op_r = sqC(sc, idxr, false)
			op_pr = op_p * op_r
			if isassigned(BQold, orbp, orbr)
				if orbp < orbr
					hnew = renormalizeright!(hnew, BQold[orbp, orbr], totensormap(-op_pr, side=:L), add_adjoint=true)
				elseif orbp == orbr
					hnew = renormalizeright!(hnew, BQold[orbp, orbr], totensormap(-op_pr, side=:L), add_adjoint=false)
				end
			end
		end
	end

	_issymmetric(hnew) || throw(ArgumentError("h matrix is not symmetric"))
	return hnew
end

renormalizestorageright(env::QCDMRGCache, site::Int, mpsj::MPSSiteTensor=env.mps[site]) = renormalizestorageright(env, site, space_r(mpsj)')
function renormalizestorageright(env::QCDMRGCache, site::Int, spacer::ElementarySpace)
	if site == length(env)
		return renormalizestorageright(env.ham, spacer)
	else
		return renormalizestorageright(storage(env, site+1), env.ham, site, spacer)
	end
end

function renormalizestorageright(ham::MolecularHamiltonian, spacer::ElementarySpace) 
	L = length(ham)
	site = L
	h1e, h2e = ham.h1e, ham.h2e
	sl, sc, sr = get_splitting(L, site)
	nl, nr = get_nl_nr(L, site)

	hnew = renormalizeHright(ham, spacer)

	A = ratensortype(spacetype(hnew), storagetype(hnew))
	id_right = isomorphism(storagetype(hnew), spacer, spacer)

	PAnew = Matrix{A}(undef, 2, 2)
	for (idxr, orbr) in enumerate(sc)
		op_r = sqC(sc, idxr, false)
		for (idxs, orbs) in enumerate(sc)
			op_s = sqC(sc, idxs, false)
			if orbr < orbs
				op_rs = op_r * op_s
				PAnew[idxr, idxs] = renormalizeright(id_right, totensormap(op_rs, side=:R))
			end
		end
	end
	BQnew = Matrix{A}(undef, nl, nl)
	for (idxp, orbp) in enumerate(sl)
		for (idxr, orbr) in enumerate(sl)
			if orbp <= orbr
				op_qs = scratch_empty()
				for (idxq, orbq) in enumerate(sc)
					op_q = sqC(sc, idxq, true)
					for (idxs, orbs) in enumerate(sc)
						op_s = sqC(sc, idxs, false)
						op_qs += h2e[orbp, orbq, orbr, orbs] * op_q * op_s
					end
				end
				if !iszero(op_qs)
					BQnew[orbp, orbr] = renormalizeright(id_right, totensormap(op_qs, side=:R))
				end
			end
		end
	end
	aTnew = Vector{A}(undef, nl)
	for (idxp, orbp) in enumerate(sl)
		op_qrs = scratch_empty()
		for (idxq, orbq) in enumerate(sc)
			op_q = sqC(sc, idxq, true)
			op_qrs += h1e[orbp, orbq] * sqC(sc, idxq, false)
			for (idxr, orbr) in enumerate(sc)
				op_r = sqC(sc, idxr, false)
				for (idxs, orbs) in enumerate(sc)
					op_s = sqC(sc, idxs, false)
					if orbr < orbs
						op_qrs += h2e[orbp, orbq, orbr, orbs] * op_q * op_r * op_s
					end
				end
			end
		end
		if !iszero(op_qrs)
			aTnew[orbp] = renormalizeright(id_right, totensormap(op_qrs, side=:R))
		end
	end

	Tanew = Vector{A}(undef, 2)
	for (idxs, orbs) in enumerate(sc)
		op_s = sqC(sc, idxs, false)
		Tanew[idxs] = renormalizeright(id_right, totensormap(op_s, side=:R))
	end

	return QCSiteStorages(hnew, BQnew, PAnew, aTnew, Tanew)
end

function renormalizestorageright(storage_old::QCSiteStorages, ham::MolecularHamiltonian, site::Int, spacer::ElementarySpace) 
	L = length(ham)
	(2 <= site < L) || throw(BoundsError())
	Hold, BQold, PAold, adagTold, Tdagaold = storage_old.H, storage_old.BQ, storage_old.PA, storage_old.adagT, storage_old.Tdaga
	h1e, h2e = ham.h1e, ham.h2e
	sl, sc, sr = get_splitting(L, site)
	nl, nr = get_nl_nr(L, site)

	A = ratensortype(spacetype(Hold), storagetype(Hold))
	id_right = isomorphism(storagetype(Hold), spacer, spacer)

	@assert length(adagTold) == nl + 2
	@assert length(Tdagaold) == nr 
	@assert size(PAold, 1) == nr
	@assert size(BQold, 1) == nl + 2


	# update A storage
	PAnew = Matrix{A}(undef, nr+2, nr+2)
	for (idxr, orbr) in enumerate(sr)
		for (idxs, orbs) in enumerate(sr)
			if idxr < idxs
				PAnew[idxr+2, idxs+2] = renormalizeright(PAold[idxr, idxs], nothing)
			end
		end
	end
	for (idxr, orbr) in enumerate(sc)
		op_r = sqC(sc, idxr, false) * sgnC(sc)
		for (idxs, orbs) in enumerate(sr)
			PAnew[idxr, idxs+2] = renormalizeright(Tdagaold[idxs], totensormap(op_r, side=:R))
		end
	end
	for (idxr, orbr) in enumerate(sc)
		op_r = sqC(sc, idxr, false)
		for (idxs, orbs) in enumerate(sc)
			op_s = sqC(sc, idxs, false)
			if orbr < orbs
				op_rs = op_r * op_s
				PAnew[idxr, idxs] = renormalizeright(id_right, totensormap(op_rs, side=:R))
			end
			
		end
	end
	# update B storage
	BQnew = Matrix{A}(undef, nl, nl)
	for (idxp, orbp) in enumerate(sl)
		for (idxr, orbr) in enumerate(sl)
			if orbp <= orbr
				if isassigned(BQold, orbp, orbr)
					BQnew[orbp, orbr] = renormalizeright(BQold[orbp, orbr], nothing)
				end
				op_qs = scratch_empty()
				for (idxq, orbq) in enumerate(sc)
					op_q = sqC(sc, idxq, true)
					for (idxs, orbs) in enumerate(sc)
						op_s = sqC(sc, idxs, false)
						op_qs += h2e[orbp, orbq, orbr, orbs] * op_q * op_s
					end
				end
				if !iszero(op_qs)
					if isassigned(BQnew, orbp, orbr)
						BQnew[orbp, orbr] = renormalizeright!(BQnew[orbp, orbr], id_right, totensormap(op_qs, side=:R))
					else
						BQnew[orbp, orbr] = renormalizeright(id_right, totensormap(op_qs, side=:R))
					end
				end

				for (idxq, orbq) in enumerate(sc)
					op_q = sqC(sc, idxq, true)
					for (idxs, orbs) in enumerate(sr)
						coef = h2e[orbp, orbq, orbr, orbs]
						if !iszero(coef)
							tmp = totensormap(coef * op_q * sgnC(sc), side=:R)
							if isassigned(BQnew, orbp, orbr)
								BQnew[orbp, orbr] = renormalizeright!(BQnew[orbp, orbr], Tdagaold[idxs], tmp)
							else									
								BQnew[orbp, orbr] = renormalizeright(Tdagaold[idxs], tmp)
							end
						end
					end
				end
				for (idxq, orbq) in enumerate(sr)
					for (idxs, orbs) in enumerate(sc)
						op_s = sqC(sc, idxs, false)
						coef = h2e[orbp, orbq, orbr, orbs]
						if (!iszero(coef)) && (dim(Tdagaold[idxq]) != 0)
							tmp = totensormap(-coef * op_s * sgnC(sc), side=:R)
							if isassigned(BQnew, orbp, orbr)
								# BQnew[orbp, orbr] = renormalizeright!(BQnew[orbp, orbr], phy_dagger(Tdagaold[idxq]), tmp)
								BQnew[orbp, orbr] = renormalizeright!(BQnew[orbp, orbr], Tdagaold[idxq], tmp, dagger=true)
							else
								BQnew[orbp, orbr] = renormalizeright(phy_dagger(Tdagaold[idxq]), tmp)
								###*** this is a bug when dagger = true?
								# BQnew[orbp, orbr] = renormalizeright(Tdagaold[idxq], tmp, dagger=true)
							end
						end
					end
				end
			end
		end
	end

	# update aT storage
	aTnew = Vector{A}(undef, nl)
	for (idxp, orbp) in enumerate(sl)
		if isassigned(adagTold, idxp)
			aTnew[idxp] = renormalizeright(adagTold[idxp], nothing)
		end

		op_qrs = scratch_empty()
		for (idxq, orbq) in enumerate(sc)
			op_q = sqC(sc, idxq, true)
			op_qrs += h1e[orbp, orbq] * sqC(sc, idxq, false)
			for (idxr, orbr) in enumerate(sc)
				op_r = sqC(sc, idxr, false)
				op_qr = op_q * op_r
				for (idxs, orbs) in enumerate(sc)
					op_s = sqC(sc, idxs, false)
					if orbr < orbs
						op_qrs += h2e[orbp, orbq, orbr, orbs] * op_qr * op_s
					end
				end
			end
		end
		if !iszero(op_qrs)
			if isassigned(aTnew, idxp)
				aTnew[idxp] = renormalizeright!(aTnew[idxp], id_right, totensormap(op_qrs, side=:R))
			else
				aTnew[idxp] = renormalizeright(id_right, totensormap(op_qrs, side=:R))
			end
		end

		for (idxq, orbq) in enumerate(sc)
			op_q = sqC(sc, idxq, true)
			for (idxr, orbr) in enumerate(sr)
				for (idxs, orbs) in enumerate(sr)
					if orbr < orbs
						coef = h2e[orbp, orbq, orbr, orbs]
						if !iszero(coef)
							if isassigned(aTnew, idxp)
								aTnew[idxp] = renormalizeright!(aTnew[idxp], PAold[idxr, idxs], totensormap(coef * op_q, side=:R))
							else
								aTnew[idxp] = renormalizeright(PAold[idxr, idxs], totensormap(coef * op_q, side=:R))
							end
						end
					end
				end
			end
		end
		for (idxq, orbq) in enumerate(sr)
			op_rs = scratch_empty()
			for (idxr, orbr) in enumerate(sc)
				op_r = sqC(sc, idxr, false)
				for (idxs, orbs) in enumerate(sc)
					if orbr < orbs
						op_s = sqC(sc, idxs, false)
						op_rs += h2e[orbp, orbq, orbr, orbs] * op_r * op_s * sgnC(sc)
					end
				end
			end
			if (!iszero(op_rs)) && (dim(Tdagaold[idxq]) != 0)
				if isassigned(aTnew, idxp)
					# aTnew[idxp] = renormalizeright!(aTnew[idxp], phy_dagger(Tdagaold[idxq]), totensormap(op_rs, side=:R))
					aTnew[idxp] = renormalizeright!(aTnew[idxp], Tdagaold[idxq], totensormap(op_rs, side=:R), dagger=true)
				else
					# aTnew[idxp] = renormalizeright(phy_dagger(Tdagaold[idxq]), totensormap(op_rs, side=:R))
					aTnew[idxp] = renormalizeright(Tdagaold[idxq], totensormap(op_rs, side=:R), dagger=true)
				end
			end
		end
		for (idxs, orbs) in enumerate(sr)
			for (idxq, orbq) in enumerate(sc)
				op_qr = scratch_empty()
				op_q = sqC(sc, idxq, true)
				for (idxr, orbr) in enumerate(sc)
					op_r = sqC(sc, idxr, false)
					op_qr += h2e[orbp, orbq, orbr, orbs] * op_q * op_r * sgnC(sc)
				end
				if !iszero(op_qr)
					if isassigned(aTnew, idxp)
						aTnew[idxp] = renormalizeright!(aTnew[idxp], Tdagaold[idxs], totensormap(op_qr, side=:R))
					else
						aTnew[idxp] = renormalizeright(Tdagaold[idxs], totensormap(op_qr, side=:R))
					end
				end
			end
		end
		for (idxr, orbr) in enumerate(sc)
			op_r = sqC(sc, idxr, false)
			tmp = -totensormap(op_r, side=:R)
			if isassigned(BQold, orbp, orbr)
				if orbp < orbr
					if isassigned(aTnew, idxp)
						aTnew[idxp] = renormalizeright!(aTnew[idxp], BQold[orbp, orbr], tmp)
					else
						aTnew[idxp] = renormalizeright(BQold[orbp, orbr], tmp)
					end
					# aTnew[idxp] = renormalizeright!(aTnew[idxp], BQold[orbp, orbr], phy_dagger(tmp), dagger=true)
				elseif orbp == orbr
					if isassigned(aTnew, idxp)
						aTnew[idxp] = renormalizeright!(aTnew[idxp], BQold[orbp, orbr], tmp)
					else
						aTnew[idxp] = renormalizeright(BQold[orbp, orbr], tmp)
					end
				end
			end
		end
	end
	# update Ta storage
	Tanew = Vector{A}(undef, nr + 2)
	for (idxs, orbs) in enumerate(sr)
		Tanew[idxs + 2] = renormalizeright(Tdagaold[idxs], nothing)
	end
	for (idxs, orbs) in enumerate(sc)
		op_s = sqC(sc, idxs, false)
		Tanew[idxs] = renormalizeright(id_right, totensormap(op_s, side=:R))
	end

	hnew = renormalizeHright(storage_old, ham, site, spacer)
	return QCSiteStorages(hnew, BQnew, PAnew, aTnew, Tanew)
end

function updatestoragerenormalizeright(storages::QCSiteStorages, mpsj)
	workspace = scratch_workspace!(mpsj)
	hnewr, BQnewr, PAnewr, aTnewr, Tanewr = storages.H, storages.BQ, storages.PA, storages.adagT, storages.Tdaga
	BQnew = _updateright_all(BQnewr, mpsj, workspace)
	PAnew = _updateright_all(PAnewr, mpsj, workspace)
	aTnew = _updateright_all(aTnewr, mpsj, workspace)
	Tanew = _updateright_all(Tanewr, mpsj, workspace)
	hnew = updaterenormalizeright(hnewr, mpsj, mpsj, workspace)	
	return QCSiteStorages(hnew, BQnew, PAnew, aTnew, Tanew)
end
updatestorageright(env::QCDMRGCache, site::Int, mpsj::MPSSiteTensor=env.mps[site]) = updatestoragerenormalizeright(renormalizestorageright(env, site, mpsj), mpsj)


function _updateright_all(storages::Vector, mpsj, workspace::Vector)
	A = mpstensortype(spacetype(mpsj), storagetype(mpsj))
	r = Vector{A}(undef, size(storages))
	indices = Int[]
	for i in eachindex(storages)
		isassigned(storages, i) && push!(indices, i)
	end
	if Threads.nthreads() > 1 && length(indices) >= MIN_RENORM_TASKS_FOR_THREADS
		Threads.@threads for i in indices
			ws = scratch_workspace!(mpsj)
			r[i] = updaterenormalizeright(storages[i], mpsj, mpsj, ws)
		end
	else
		for i in indices
			r[i] = updaterenormalizeright(storages[i], mpsj, mpsj, workspace)
		end
	end
	return r
end
function _updateright_all(storages::Matrix, mpsj, workspace::Vector)
	A = mpstensortype(spacetype(mpsj), storagetype(mpsj))
	r = Matrix{A}(undef, size(storages))
	n = size(storages, 1)
	tasks = NTuple{2,Int}[]
	for i in 1:n, j in i:n
		isassigned(storages, i, j) && push!(tasks, (i, j))
	end
	if Threads.nthreads() > 1 && length(tasks) >= MIN_RENORM_TASKS_FOR_THREADS
		Threads.@threads for (i, j) in tasks
			ws = scratch_workspace!(mpsj)
			r[i, j] = updaterenormalizeright(storages[i, j], mpsj, mpsj, ws)
		end
	else
		for (i, j) in tasks
			r[i, j] = updaterenormalizeright(storages[i, j], mpsj, mpsj, workspace)
		end
	end
	return r
end

renormalizeHleft(env::QCDMRGCache, site::Int, mpsj::MPSTensor) = renormalizeHleft(env, site, space_l(mpsj))
function renormalizeHleft(env::QCDMRGCache, site::Int, spacel::ElementarySpace)
	if site == 1
		return renormalizeHleft(env.ham, spacel)
	else
		return renormalizeHleft(storage(env, site-1), env.ham, site, spacel)
	end
end

function updateHleft(env::QCDMRGCache, site::Int, mpsj::MPSTensor=env.mps[site])
	hnew = renormalizeHleft(env, site, mpsj)
	return updaterenormalizeleft(hnew, mpsj, mpsj)
end

function renormalizeHleft(ham::MolecularHamiltonian, spacel::ElementarySpace)
	hj = hlocal(ham, 1)
	id_left = isomorphism(storagetype(hj), spacel, spacel)
	hnew = renormalizeleft(id_left, hj)
	_issymmetric(hnew) || throw(ArgumentError("h matrix is not symmetric"))
	return hnew
end

function renormalizeHleft(storage_old::QCSiteStorages, ham::MolecularHamiltonian, site::Int, spacel::ElementarySpace) 
	L = length(ham)
	(1 < site <= L) || throw(BoundsError())
	Hold, BQold, PAold, adagTold, Tdagaold = storage_old.H, storage_old.BQ, storage_old.PA, storage_old.adagT, storage_old.Tdaga
	h1e, h2e = ham.h1e, ham.h2e
	sl, sc, sr = get_splitting(L, site)
	nl, nr = get_nl_nr(L, site)

	id_left = isomorphism(storagetype(Hold), spacel, spacel)
	hnew = renormalizeleft(id_left, hlocal(ham, site))

	@assert length(adagTold) == nl
	@assert length(Tdagaold) == nr + 2
	@assert size(PAold, 1) == nr + 2
	@assert size(BQold, 1) == nl

	hnew = renormalizeleft!(hnew, Hold, isomorphism(_u1u1_pspace, _u1u1_pspace))
	# eat aT
	for (idxp, orbp) in enumerate(sl)
		op_qrs = scratch_empty()
		for (idxq, orbq) in enumerate(sc)
			op_q = sqC(sc, idxq, true)
			op_qrs += h1e[orbp, orbq] * sqC(sc, idxq, false)
			for (idxr, orbr) in enumerate(sc)
				op_r = sqC(sc, idxr, false)
				for (idxs, orbs) in enumerate(sc)
					op_s = sqC(sc, idxs, false)
					if orbr < orbs
						op_qrs += h2e[orbp, orbq, orbr, orbs] * op_q * op_r * op_s
					end
				end
			end
		end
		if !iszero(op_qrs)
			hnew = renormalizeleft!(hnew, adagTold[orbp], totensormap(op_qrs, side=:R), add_adjoint=true)
		end
	end
	# eat Ta
	for (idxs, orbs) in enumerate(sc)
		op_s = sqC(sc, idxs, false)
		if isassigned(Tdagaold, idxs)
			hnew = renormalizeleft!(hnew, Tdagaold[idxs], totensormap(op_s, side=:R), add_adjoint=true)
		end
	end
	# eat A
	for (idxr, orbr) in enumerate(sc)
		op_r = sqC(sc, idxr, false)
		for (idxs, orbs) in enumerate(sc)
			op_s = sqC(sc, idxs, false)
			if (orbr < orbs) && isassigned(PAold, idxr, idxs)
				op_rs = op_r * op_s
				hnew = renormalizeleft!(hnew, PAold[idxr, idxs], totensormap(op_rs, side=:R), add_adjoint=true)
			end
		end
	end
	# eat B
	for (idxp, orbp) in enumerate(sl)
		for (idxr, orbr) in enumerate(sl)
			op_qs = scratch_empty()
			for (idxq, orbq) in enumerate(sc)
				op_q = sqC(sc, idxq, true)
				for (idxs, orbs) in enumerate(sc)
					op_s = sqC(sc, idxs, false)
					op_qs -= h2e[orbp, orbq, orbr, orbs] * op_q * op_s
				end
			end
			if !iszero(op_qs)
				tmp = totensormap(op_qs, side=:R)
				if orbp < orbr
					hnew = renormalizeleft!(hnew, BQold[orbp, orbr], tmp, add_adjoint=true)
				elseif orbp == orbr
					hnew = renormalizeleft!(hnew, BQold[orbp, orbr], tmp, add_adjoint=false)
				end
			end

		end
	end

	_issymmetric(hnew) || throw(ArgumentError("h matrix is not symmetric"))
	return hnew
end

renormalizestorageleft(env::QCDMRGCache, site::Int, mpsj::MPSTensor=env.mps[site]) = renormalizestorageleft(env, site, space_l(mpsj))
function renormalizestorageleft(env::QCDMRGCache, site::Int, spacel::ElementarySpace)
	if site == 1
		return renormalizestorageleft(env.ham, spacel)
	else
		return renormalizestorageleft(storage(env, site-1), env.ham, site, spacel)
	end
end

function renormalizestorageleft(ham::MolecularHamiltonian, spacel::ElementarySpace)
	L = length(ham)
	site = 1
	h1e, h2e = ham.h1e, ham.h2e
	sl, sc, sr = get_splitting(L, site)
	nl, nr = get_nl_nr(L, site)

	hnew = renormalizeHleft(ham, spacel)
	A = ratensortype(spacetype(hnew), storagetype(hnew))
	id_left = isomorphism(storagetype(hnew), spacel, spacel)
	ops = SiteOps(sc)

	PAnew = Matrix{A}(undef, nr, nr)
	for (idxr, orbr) in enumerate(sr)
		for (idxs, orbs) in enumerate(sr)
			if orbr < orbs
				op_pq = h2e_pair_op_pq(sc, h2e, orbr, orbs, ops)
				if !iszero(op_pq)
					PAnew[idxr, idxs] = renormalizeleft(id_left, totensormap(op_pq, side=:L))
				end
			end
		end
	end

	BQnew = Matrix{A}(undef, 2, 2)
	for (idxp, orbp) in enumerate(sc)
		op_p = sqC(sc, idxp, true)
		for (idxr, orbr) in enumerate(sc)
			if orbp <= orbr
				op_r = sqC(sc, idxr, false)
				op_pr = op_p * op_r
				BQnew[idxp, idxr] = renormalizeleft(id_left, totensormap(op_pr, side=:L))					
			end
		end
	end

	aTnew = Vector{A}(undef, 2)
	for (idxp, orbp) in enumerate(sc)
		op_p = sqC(sc, idxp, true) * sgnC(sc)
		aTnew[idxp] = renormalizeleft(id_left, totensormap(op_p, side=:L))
	end

	Tanew = Vector{A}(undef, nr)
	for (idxs, orbs) in enumerate(sr)
		op_pqr = scratch_empty()
		for (idxp, orbp) in enumerate(sc)
			op_p = sqC(sc, idxp, true)
			for (idxq, orbq) in enumerate(sc)
				op_q = sqC(sc, idxq, true)
				if orbp < orbq
					op_pq = op_p * op_q
					for (idxr, orbr) in enumerate(sc)
						op_r = sqC(sc, idxr, false)
						op_pqr += h2e[orbp, orbq, orbr, orbs] * op_pq * op_r
					end
				end
				
			end
		end
		if !iszero(op_pqr)
			op_pqr = op_pqr * sgnC(sc)
			Tanew[idxs] = renormalizeleft(id_left, totensormap(op_pqr, side=:L)) 
		end
	end

	return QCSiteStorages(hnew, BQnew, PAnew, aTnew, Tanew)
end

function renormalizestorageleft(storage_old::QCSiteStorages, ham::MolecularHamiltonian, site::Int, spacel::ElementarySpace)
	L = length(ham)
	(1 < site <= L-1) || throw(BoundsError())
	Hold, BQold, PAold, adagTold, Tdagaold = storage_old.H, storage_old.BQ, storage_old.PA, storage_old.adagT, storage_old.Tdaga
	h1e, h2e = ham.h1e, ham.h2e
	sl, sc, sr = get_splitting(L, site)
	nl, nr = get_nl_nr(L, site)

	A = ratensortype(spacetype(Hold), storagetype(Hold))
	id_left = isomorphism(storagetype(Hold), spacel, spacel)

	@assert length(adagTold) == nl
	@assert length(Tdagaold) == nr + 2
	@assert size(PAold, 1) == nr + 2
	@assert size(BQold, 1) == nl

	# update PA storage
	PAnew = Matrix{A}(undef, nr, nr)
	fill_PA_left!(PAnew, PAold, adagTold, id_left, sr, sc, sl, h2e)

	# update BQ storage
	# only upper triangular of B is stored
	BQnew = Matrix{A}(undef, nl+2, nl+2)
	for (idxp, orbp) in enumerate(sl)
		for (idxr, orbr) in enumerate(sl)
			if orbp <= orbr
				BQnew[orbp, orbr] = renormalizeleft(BQold[idxp, idxr], nothing)
			end
		end
	end
	for (idxp, orbp) in enumerate(sc)
		op_p = sqC(sc, idxp, true)
		for (idxr, orbr) in enumerate(sc)
			if orbp <= orbr
				op_r = sqC(sc, idxr, false)
				op_pr = op_p * op_r
				BQnew[orbp, orbr] = renormalizeleft(id_left, totensormap(op_pr, side=:L))
			end
		end
	end		
	for (idxp, orbp) in enumerate(sl)
		for (idxr, orbr) in enumerate(sc)
			op_r = sqC(sc, idxr, false)
			BQnew[orbp, orbr] = renormalizeleft(adagTold[idxp], totensormap(op_r, side=:L))
		end
	end

	# update aT storage
	aTnew = Vector{A}(undef, nl+2)
	for (idxp, orbp) in enumerate(sl)
		aTnew[orbp] = renormalizeleft(adagTold[idxp], nothing)
	end
	for (idxp, orbp) in enumerate(sc)
		op_p = sqC(sc, idxp, true) * sgnC(sc)
		aTnew[orbp] = renormalizeleft(id_left, totensormap(op_p, side=:L))
	end
	# update Ta storage
	Tanew = Vector{A}(undef, nr)
	for (idxs, orbs) in enumerate(sr)
		if isassigned(Tdagaold, idxs+2)
			Tanew[idxs] = renormalizeleft(Tdagaold[idxs+2], nothing)
		end
	end
	for (idxs, orbs) in enumerate(sr)
		op_pqr = scratch_empty()
		for (idxp, orbp) in enumerate(sc)
			op_p = sqC(sc, idxp, true)
			for (idxq, orbq) in enumerate(sc)
				op_q = sqC(sc, idxq, true)
				if orbp < orbq
					op_pq = op_p * op_q
					for (idxr, orbr) in enumerate(sc)
						op_r = sqC(sc, idxr, false)
						op_pqr += h2e[orbp, orbq, orbr, orbs] * op_pq * op_r
					end
				end
			end
		end
		if !iszero(op_pqr)
			op_pqr = op_pqr * sgnC(sc)
			if isassigned(Tanew, idxs)
				Tanew[idxs] = renormalizeleft!(Tanew[idxs], id_left, totensormap(op_pqr, side=:L))
			else
				Tanew[idxs] = renormalizeleft(id_left, totensormap(op_pqr, side=:L))
			end
		end
	end
	for (idxs, orbs) in enumerate(sr)
		for (idxr, orbr) in enumerate(sc)
			op_r = sqC(sc, idxr, false) * sgnC(sc)
			if isassigned(PAold, idxr, idxs+2)
				if isassigned(Tanew, idxs)
					Tanew[idxs] = renormalizeleft!(Tanew[idxs], PAold[idxr, idxs+2], totensormap(op_r, side=:L))
				else
					Tanew[idxs] = renormalizeleft(PAold[idxr, idxs+2], totensormap(op_r, side=:L))
				end
			end
		end

		for (idxr, orbr) in enumerate(sl)
			op_qp = scratch_empty()
			for (idxp, orbp) in enumerate(sc)
				op_p = sqC(sc, idxp, false)
				for (idxq, orbq) in enumerate(sc)
					op_q = sgnC(sc) * sqC(sc, idxq, false) 
					if orbp < orbq
						op_qp -= h2e[orbp, orbq, orbr, orbs] * op_q * op_p
					end
				end
			end
			# this part is tricky and needs to be modified
			if !iszero(op_qp)
				# tmp = renormalizeleft(adagTold[idxr], totensormap(op_qp, side=:L))
				# tmp2 = phy_dagger(tmp)
				if isassigned(Tanew, idxs)
					# Tanew[idxs] += tmp2
					Tanew[idxs] = renormalizeleft_odagger!(Tanew[idxs], adagTold[idxr], totensormap(op_qp, side=:L))
				else
					# Tanew[idxs] = tmp2
					Tanew[idxs] = renormalizeleft_odagger(adagTold[idxr], totensormap(op_qp, side=:L))
				end
			end
		end
		for (idxp, orbp) in enumerate(sl)
			op_qr = scratch_empty()
			for (idxq, orbq) in enumerate(sc)
				op_q = sqC(sc, idxq, true)
				for (idxr, orbr) in enumerate(sc)
					op_r = sqC(sc, idxr, false) * sgnC(sc)
					op_qr += h2e[orbp, orbq, orbr, orbs] * op_q * op_r
				end
			end
			if !iszero(op_qr)
				if isassigned(Tanew, idxs)
					Tanew[idxs] = renormalizeleft!(Tanew[idxs], adagTold[idxp], totensormap(op_qr, side=:L))
				else
					Tanew[idxs] = renormalizeleft(adagTold[idxp], totensormap(op_qr, side=:L))
				end
			end
		end

		for (idxq, orbq) in enumerate(sc)
			op_q = sqC(sc, idxq, true)
			for (idxp, orbp) in enumerate(sl)
				for (idxr, orbr) in enumerate(sl)
					coef = h2e[orbp, orbq, orbr, orbs]
					if !iszero(coef)
						tmp = totensormap(-coef * op_q * sgnC(sc), side=:L)
						if orbp < orbr
							if isassigned(Tanew, idxs)
								Tanew[idxs] = renormalizeleft!(Tanew[idxs], BQold[idxp, idxr], tmp)
							else
								Tanew[idxs] = renormalizeleft(BQold[idxp, idxr], tmp)
							end
						elseif orbp == orbr
							if isassigned(Tanew, idxs)
								Tanew[idxs] = renormalizeleft!(Tanew[idxs], BQold[idxp, idxr], tmp)
							else
								Tanew[idxs] = renormalizeleft(BQold[idxp, idxr], tmp)
							end
						else
							if isassigned(Tanew, idxs)
								# Tanew[idxs] = renormalizeleft!(Tanew[idxs], phy_dagger(BQold[idxr, idxp]), tmp)
								Tanew[idxs] = renormalizeleft!(Tanew[idxs], BQold[idxr, idxp], tmp, dagger=true)
							else
								# Tanew[idxs] = renormalizeleft(phy_dagger(BQold[idxr, idxp]), tmp)
								Tanew[idxs] = renormalizeleft(BQold[idxr, idxp], tmp, dagger=true)
							end
						end
					end
				end
			end
		end
	end

	hnew = renormalizeHleft(storage_old, ham, site, spacel)
	return QCSiteStorages(hnew, BQnew, PAnew, aTnew, Tanew)
end

function updatestoragerenormalizeleft(storages::QCSiteStorages, mpsj)
	workspace = scratch_workspace!(mpsj)
	hnewr, BQnewr, PAnewr, aTnewr, Tanewr = storages.H, storages.BQ, storages.PA, storages.adagT, storages.Tdaga

	BQnew = _updateleft_all(BQnewr, mpsj, workspace)
	PAnew = _updateleft_all(PAnewr, mpsj, workspace)
	aTnew = _updateleft_all(aTnewr, mpsj, workspace)
	Tanew = _updateleft_all(Tanewr, mpsj, workspace)
	hnew = updaterenormalizeleft(hnewr, mpsj, mpsj, workspace)	
	return QCSiteStorages(hnew, BQnew, PAnew, aTnew, Tanew)
end
updatestorageleft(env::QCDMRGCache, site::Int, mpsj::MPSTensor=env.mps[site]) = updatestoragerenormalizeleft(renormalizestorageleft(env, site, mpsj), mpsj)

function _updateleft_all(storages::Vector, mpsj, workspace::Vector)
	A = mpstensortype(spacetype(mpsj), storagetype(mpsj))
	r = Vector{A}(undef, size(storages))
	indices = Int[]
	for i in eachindex(storages)
		isassigned(storages, i) && push!(indices, i)
	end
	if Threads.nthreads() > 1 && length(indices) >= MIN_RENORM_TASKS_FOR_THREADS
		Threads.@threads for i in indices
			ws = scratch_workspace!(mpsj)
			r[i] = updaterenormalizeleft(storages[i], mpsj, mpsj, ws)
		end
	else
		for i in indices
			r[i] = updaterenormalizeleft(storages[i], mpsj, mpsj, workspace)
		end
	end
	return r
end
function _updateleft_all(storages::Matrix, mpsj, workspace::Vector)
	A = mpstensortype(spacetype(mpsj), storagetype(mpsj))
	r = Matrix{A}(undef, size(storages))
	n = size(storages, 1)
	tasks = NTuple{2,Int}[]
	for i in 1:n, j in i:n
		isassigned(storages, i, j) && push!(tasks, (i, j))
	end
	if Threads.nthreads() > 1 && length(tasks) >= MIN_RENORM_TASKS_FOR_THREADS
		Threads.@threads for (i, j) in tasks
			ws = scratch_workspace!(mpsj)
			r[i, j] = updaterenormalizeleft(storages[i, j], mpsj, mpsj, ws)
		end
	else
		for (i, j) in tasks
			r[i, j] = updaterenormalizeleft(storages[i, j], mpsj, mpsj, workspace)
		end
	end
	return r
end

# function phy_dagger(t::RATensor)
#     t′ = t'
#     return flip2(permute(t′, (1,2,5), (3,4)))
# end
# function flip2(t::RATensor)
#     vspace = space(t, 3)
#     F = isomorphism(storagetype(t), flip(vspace), vspace)
#     @tensor t2[3,4,1;5,6] := F[1,2] * t[3,4,2,5,6]
# end
# function phy_dagger(t::RATensor)
#     t′ = t'
#     vspace = space(t′, 5)
#     F = isomorphism(storagetype(t), flip(vspace), vspace)
#     @tensor r[3,4,1;5,6] := F[1,2] * t′[3,4,5,6,2]
#     # return flip2(permute(t′, (1,2,5), (3,4)))
# end

function renormalizeleft_odagger(hold::MPSTensor, mpoj::MPSTensor)
    mspace = fuse(space(mpoj, 2), space(hold, 2))       
    hnew = RATensor(zeros, scalartype(hold), space(hold, 3)' ⊗ space(mpoj, 3)' ⊗ mspace', space(hold, 1) ⊗ space(mpoj, 1))        	
    return renormalizeleft_odagger!(hnew, hold, mpoj)
end

function renormalizeleft_odagger2!(hnew::RATensor, hold::MPSTensor, mpoj::MPSTensor)
    (space(hnew, 4)' == space(hold, 1)) && (space(hnew, 5)' == space(mpoj, 1)) && 
        (space(hnew, 1)' == space(hold, 3)) && (space(hnew, 2)' == space(mpoj, 3)) || throw(SpaceMismatch())
    (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

    tmp = DMRG.loose_isometry(storagetype(hnew), space(hnew, 3)', space(hold, 2) ⊗ space(mpoj, 2) )
    @tensor hnew[3,6,7;1,4] += conj(hold[1,2,3]) * conj(mpoj[4,5,6]) * conj(tmp[7,2,5])
	return hnew
end
function renormalizeleft_odagger!(hnew::RATensor, hold::MPSTensor, mpoj::MPSTensor)
    (space(hnew, 4)' == space(hold, 1)) && (space(hnew, 5)' == space(mpoj, 1)) && 
        (space(hnew, 1)' == space(hold, 3)) && (space(hnew, 2)' == space(mpoj, 3)) || throw(SpaceMismatch())
    (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

    (space(hnew, 3)' ≅ space(hold, 2) ⊗ space(mpoj, 2)) || return hnew

    # tmp = DMRG.loose_isometry(storagetype(hnew), space(hnew, 3)', space(hold, 2) ⊗ space(mpoj, 2) )
    # @tensor hnew[3,6,7;1,4] += conj(hold[1,2,3]) * conj(mpoj[4,5,6]) * conj(tmp[7,2,5])

    for (f1l, f1r) in fusiontrees(hold)
        v = StridedView(dropdims(hold[f1l, f1r], dims=2)')
        (f1l′, f1r′), coef0 = only(permute(f1l, f1r, (1,), (2,3)))
        c1 = f1l′.coupled
        for (f2l, f2r) in fusiontrees(mpoj)
            α = only(mpoj[f2l, f2r])
            (f2l′, f2r′), coef1 = only(permute(f2l, f2r, (1,), (2,3)))
            c2 = f2l′.coupled
            c = first(c1 ⊗ c2)
            for (fl, coef2) in TK.merge(f1l′, f2l′, c)
            	for (fr, coef3) in TK.merge(f1r′, f2r′, c)
            		uncoupled = (fr.uncoupled[2], fr.uncoupled[4], first(fr.uncoupled[1] ⊗ fr.uncoupled[3]) )
            		isdual = (fr.isdual[2], fr.isdual[4], true)
            		 fr′ = FusionTree(uncoupled, fr.coupled, isdual)
            		 coef = α * coef0 * coef1 * coef2 * coef3
            		 out = sreshape(hnew[fr′, fl], size(v))
            		 # out .+= coef .* v
            		 axpy!(coef, v, out)
            	end

            end
        end
    end

	return hnew
end