push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/DMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/InfiniteDMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/GeneralHamiltonians/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/QCMPO/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/SphericalTensors/src")

using Test
using SphericalTensors
using QCMPO
using Random

# push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")
# using QCDMRG
# using QCDMRG: energy_left!, energy_right!

include(dirname(Base.@__DIR__) * "/src/includes.jl")

Random.seed!(1324)

function random_molecular_hamiltonian(L::Int)
	h1e = randn(L, L)
	h1e = h1e + h1e'
	h2e = randn(L, L, L, L)
	# h2e′ = zeros(L, L, L, L)
	h2e′ = 0.5 * (h2e + permutedims(h2e, (3,4,1,2)))
	return h1e, h2e′
end


include("groundstate.jl")
include("renormalization.jl")
