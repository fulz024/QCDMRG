"""
	struct MolecularHamiltonian

The one-body and antisymmetrized two-body coefficients
"""
struct MolecularHamiltonian
	h1e::Array{Float64, 2}
	h2e::Array{Float64, 4}

function MolecularHamiltonian(h1e::AbstractMatrix{<:Real}, h2e::AbstractArray{<:Real, 4})
	issymmetric(h1e) || throw(ArgumentError("h1e should be a symmetric matrix"))
	@assert size(h1e, 1) == size(h2e, 1) == size(h2e, 2) == size(h2e, 3) == size(h2e, 4)
	h1e, h2e = get_spin_orbitals(h1e, h2e)
	h2e = remove_antisymmetric(antisymmetrize(h2e))
	new(h1e, h2e) 
end

end
Base.length(x::MolecularHamiltonian) = div(size(x.h1e, 1), 2)

function DMRG.bond_dimension(h::MolecularHamiltonian, bond::Int)
	(1<= bond < length(h)) || throw(BoundsError())
	nl = 2 * bond
	nr = 2 * (length(h) - bond)
	# hstorage
	D = 1
	# aT
	D += 2 * nl
	# Ta
	D += 2 * nr
	# PA
	D += nr * (nr-1)
	# BQ
	D += nl * nl
	return D
end
DMRG.bond_dimensions(h::MolecularHamiltonian) = [bond_dimension(h, i) for i in 1:length(h)-1]

function hlocal(ham::MolecularHamiltonian, site::Int)
	h1e, h2e = ham.h1e, ham.h2e
	sc = (2 * site - 1, 2 * site)
	tmp = empty_operator()
	for (idxp, orbp) in enumerate(sc)
		op_p = sqC(sc, idxp, true)
		for (idxq, orbq) in enumerate(sc)
			op_q = sqC(sc, idxq, false)
			op = op_p * op_q
			tmp += h1e[orbp,orbq]*op
		end
	end
	for (idxp, orbp) in enumerate(sc)
		op_p = sqC(sc, idxp, true)
		for (idxq, orbq) in enumerate(sc)
			op_q = sqC(sc, idxq, true)
			if orbp < orbq
				op_pq = op_p * op_q
				for (idxr, orbr) in enumerate(sc)
					op_r = sqC(sc, idxr, false)
					for (idxs, orbs) in enumerate(sc)
						op_s = sqC(sc, idxs, false)
						if orbr < orbs
							op_rs = op_r * op_s
							op = op_pq * op_rs
							tmp += h2e[orbp,orbq,orbr,orbs]*op
						end
					end
				end

			end
		end
	end
	return tomatrixmap(tmp)
end

function get_splitting(L::Int, site::Int)
	sc = (2 * site - 1, 2 * site)
	sl = collect(1:2*site-2)
	sr = collect((2*site+1):2*L)
	return sl, sc, sr
end

function get_nl_nr(L::Int, site::Int)
	return 2*(site-1), 2*(L-site)
end


struct QCSiteStorages{_A, _B}
	# diagonal
	H::_B
	# off-diagonal
	BQ::Matrix{_A} # upper triangular
	PA::Matrix{_A} # upper triangular, zero diagonal
	adagT::Vector{_A}
	Tdaga::Vector{_A}
end

function renormalizedstorage(x::QCSiteStorages)
	H, BQ, PA, adagT, Tdaga = x.H, x.BQ, x.PA, x.adagT, x.Tdaga
	Hr = renormalizedoperator(H)
	A = mpstensortype(spacetype(Hr), storagetype(Hr))
	BQr = Matrix{A}(undef, size(BQ))
	for i in 1:size(BQ, 1)
		for j in i:size(BQ, 2)
			if isassigned(BQ, i, j)
				BQr[i, j] = renormalizedoperator(BQ[i, j])
			end
		end
	end
	PAr = Matrix{A}(undef, size(PA))
	for i in 1:size(PA, 1)
		for j in i:size(PA, 2)
			if isassigned(PA, i, j)
				PAr[i, j] = renormalizedoperator(PA[i, j])
			end
		end
	end	
	adagTr = Vector{A}(undef, size(adagT))
	for i in 1:length(adagT)
		if isassigned(adagT, i)
			adagTr[i] = renormalizedoperator(adagT[i])
		end
	end
	Tdagar = Vector{A}(undef, size(Tdaga))
	for i in 1:length(Tdaga)
		if isassigned(Tdaga, i)
			Tdagar[i] = renormalizedoperator(Tdaga[i])
		end
	end
	return QCSiteStorages(Hr, BQr, PAr, adagTr, Tdagar)
end

	

struct QCDMRGCache{A<:MPSTensor, B<:MPSBondTensor, _MPS}
	ham::MolecularHamiltonian
	mps::_MPS
	#-----storages-----
	# diagonal
	Hstorage::Vector{B}
	# off-diagonal
	BQstorage::Vector{Matrix{A}}
	PAstorage::Vector{Matrix{A}} # each element is an upper-triangular matrix
	adagTstorage::Vector{Vector{A}}
	Tdagastorage::Vector{Vector{A}}
end

Base.length(x::QCDMRGCache) = length(x.ham)
storage(m::QCDMRGCache, site::Int) = QCSiteStorages(m.Hstorage[site], m.BQstorage[site], m.PAstorage[site], m.adagTstorage[site], m.Tdagastorage[site])

function setstorage!(m::QCDMRGCache, site::Int, storages::QCSiteStorages)
	m.Hstorage[site] = storages.H
	m.BQstorage[site] = storages.BQ
	m.PAstorage[site] = storages.PA
	m.adagTstorage[site] = storages.adagT
	m.Tdagastorage[site] = storages.Tdaga
	return m
end

function updatestorageright!(env::QCDMRGCache, site::Int, mpsj::MPSTensor=env.mps[site]) 
	storage_new = updatestorageright(env, site, permute(mpsj, (1,), (2,3)))
	return setstorage!(env, site, storage_new)
end
function updatestorageleft!(env::QCDMRGCache, site::Int, mpsj::MPSTensor=env.mps[site])
	storage_new = updatestorageleft(env, site, mpsj)
	return setstorage!(env, site, storage_new)
end

function energy_left!(env::QCDMRGCache)
	leftorth!(env.mps)
	for site in 1:length(env)-1
		updatestorageleft!(env, site)
	end
	hnew = updateHleft(env, length(env))
	return tr(hnew)
end
function energy_right!(env::QCDMRGCache)
	rightorth!(env.mps)
	for site in length(env):-1:2
		updatestorageright!(env, site)
	end
	hnew = updateHright(env, 1)
	return tr(hnew)
end

function DMRG.environments(ham::MolecularHamiltonian, mps::MPS{A}) where {A}
	rightorth!(mps)
	(length(ham) == length(mps)) || throw(DimensionMismatch())
	L = length(ham)
	B = bondtensortype(spacetype(mps), Matrix{scalartype(mps)})
	Hstorage = Vector{B}(undef, L)
	BQstorage = Vector{Matrix{A}}(undef, L)
	PAstorage = Vector{Matrix{A}}(undef, L)
	adagTstorage = Vector{Vector{A}}(undef, L)
	Tdagastorage = Vector{Vector{A}}(undef, L)

	env = QCDMRGCache(ham, mps, Hstorage, BQstorage, PAstorage, adagTstorage, Tdagastorage)
	for site in L:-1:2
		updatestorageright!(env, site)
	end
	return env
end

function DMRG.expectation(ham::MolecularHamiltonian, mps::MPS)
	env = environments(ham, copy(mps))
	return energy_right!(env)
end

include("renormalizestorage.jl")
include("heff2.jl")
include("davidson.jl")
include("dmrg2.jl")