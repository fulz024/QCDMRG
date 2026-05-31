println("------------------------------------")
println("|        Ground state energy       |")
println("------------------------------------")

using JSON

function do_dmrg(env)
	# Tighter bond eigsolve than production default (1e-5) for ED comparison at 1e-8.
	alg = QCDMRG2(noise=1.0e-10, toleig=1.0e-8)
	_energies1 = Float64[]
	for n in 1:5
		append!(_energies1, sweep!(env, alg)[1])
	end
	return _energies1[end]
end

@testset "test ground state energies" begin

	for L in 2:5
		println("random DMRG test with L = ", L)
		h1e, h2e = random_molecular_hamiltonian(L)
		ham = MolecularHamiltonian(h1e, h2e)

		mps = randomqcmps(L; D=20, right=Rep[U₁×U₁]((div(L,2), div(L+1, 2))=>1))
		env = environments(ham, copy(mps))

		_energy1 = energy_left!(env)
		_energy2 = energy_right!(env)

		tol = 1.0e-12

		@test abs(_energy1 - _energy2) < tol

		mpo = qcmpo(h1e, h2e, ChargeCharge())
		_energy3 = expectation(mpo, mps)

		@test abs(_energy1 - _energy3) < tol

		_energy1 = do_dmrg(env)

		eigvalue2, eigvector = ground_state(mpo, ED(), right=space_r(mps)')

		@test abs(_energy1 - eigvalue2) < 1.0e-8

	end

	for L in 2:7
		h1e, h2e = random_molecular_hamiltonian(L)
		h2e .= 0
		ham = MolecularHamiltonian(h1e, h2e)

		mps = randomqcmps(L; D=20, right=Rep[U₁×U₁]((div(L,2), div(L+1, 2))=>1))
		env = environments(ham, copy(mps))

		_energy1 = energy_left!(env)
		_energy2 = energy_right!(env)

		tol = 1.0e-12

		@test abs(_energy1 - _energy2) < tol
	end	

end


function read_data(pathname)
	data = JSON.parsefile(pathname)
	E0 = data["E0"]
	L = data["L"]
	t = data["t"]
	t = [t...]
	v = data["v"]
	v = [v...]
	return E0, reshape(t, (L, L)), reshape(v, (L, L, L, L))
end

read_lih_data() = read_data(joinpath(@__DIR__, "lih.json"))

const LiH_FCI_ENERGY = -7.78446028003123

@testset "LiH ground state with ED" begin
	E0, t, v = read_lih_data()

	ham = MolecularHamiltonian(0.5 * (t + t'), 0.5 * v)
	L = length(ham)
	mps = randomqcmps(L; D=20, right=Rep[U₁×U₁]((2, 2)=>1))
	env = environments(ham, mps)
	eigvalue = do_dmrg(env)

	energy = eigvalue + E0
	@test abs(energy - LiH_FCI_ENERGY) < 1.0e-8
end

