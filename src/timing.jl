using Printf

"""
Per half-sweep timings.

- `Teff`: assemble `QCCenter` (`renormalizedstorage` + `terms`) and `qc_diagonal_aa!` when preconditioning is on
- `Teig`: Davidson (default, Olsen) / Lanczos — `H|ψ⟩` matvecs at bond eigsolve (`nmv`)
- `Tmve`: environment shift (`renormalizestorage*`) + post-SVD `updatestoragerenormalize*` / `setstorage!`
- `Tsvd`: two-site SVD truncation
- `Tsplt`: split / merge MPS after SVD (normalize, bond tensors, energy check)

`BondEigRecord`: per-bond eigsolve stats (`nmv` = solver matvecs).
"""
struct BondEigRecord
	bond::Int
	energy::Float64
	normres::Float64
	nmv::Int
	numiter::Int
end

# `toleig`: stop when ||Hψ - Eψ|| < toleig (same as KrylovKit `Lanczos.tol`).
bond_eig_resnorm(rec::BondEigRecord) = rec.normres

mutable struct DMRGTiming
	teff::Float64
	teig::Float64
	tmve::Float64
	tmve_heavy::Float64
	tmve_light::Float64
	tsvd::Float64
	tsplt::Float64
	nbonds::Int
	n_matvec::Int
	bond_records::Vector{BondEigRecord}
end

DMRGTiming() = DMRGTiming(0, 0, 0, 0, 0, 0, 0, 0, 0, BondEigRecord[])

function reset!(t::DMRGTiming)
	t.teff = t.teig = t.tmve = 0.0
	t.tmve_heavy = t.tmve_light = 0.0
	t.tsvd = t.tsplt = 0.0
	t.nbonds = 0
	t.n_matvec = 0
	empty!(t.bond_records)
	return t
end

function total_sweep_time(t::DMRGTiming)
	return t.teff + t.teig + t.tmve + t.tsvd + t.tsplt
end

struct DMRGSweepTiming
	forward::DMRGTiming
	backward::DMRGTiming
end

DMRGSweepTiming() = DMRGSweepTiming(DMRGTiming(), DMRGTiming())

const _dmrg_timing = Ref{Union{Nothing, DMRGTiming}}(nothing)

active_dmrg_timing() = _dmrg_timing[]

function reset_dmrg_timing!()
	_dmrg_timing[] = DMRGTiming()
	return _dmrg_timing[]
end

function finish_dmrg_timing!()
	t = _dmrg_timing[]
	_dmrg_timing[] = nothing
	return t
end

function print_dmrg_timing(t::DMRGTiming; io::IO=stdout, prefix::String="", direction::String="")
	dir = isempty(direction) ? "" : " | Direction = $direction"
	@printf(io, "%sTime sweep = %8.3f%s\n", prefix, total_sweep_time(t), dir)
	@printf(io, "%s | Teff = %.3f | Teig = %.3f | Tmve = %.3f | Tsvd = %.3f | Tsplt = %.3f",
		prefix, t.teff, t.teig, t.tmve, t.tsvd, t.tsplt)
	if t.tmve_heavy > 0 || t.tmve_light > 0
		@printf(io, " | Tmve_h = %.3f | Tmve_l = %.3f", t.tmve_heavy, t.tmve_light)
	end
	if t.nbonds > 0
		@printf(io, " | nbonds = %d", t.nbonds)
	end
	if t.n_matvec > 0
		@printf(io, " | n_mv = %d", t.n_matvec)
	end
	println(io)
	return t
end

function print_dmrg_sweep_timing(st::DMRGSweepTiming; io::IO=stdout, prefix::String="")
	print_dmrg_timing(st.forward; io=io, prefix=prefix, direction="forward")
	print_dmrg_timing(st.backward; io=io, prefix=prefix, direction="backward")
	return st
end

function print_bond_eig_record(rec::BondEigRecord; io::IO=stdout, prefix::String="", direction::String="", D::Int=0, tol::Union{Real,Nothing}=nothing)
	arrow = direction == "backward" ? "<--" : "-->"
	b1, b2 = rec.bond - 1, rec.bond
	if tol === nothing
		@printf(io, "%s%s bond = %2d-%2d .. D = %4d nmv = %4d E = % .10f resnorm = %.2e\n",
			prefix, arrow, b1, b2, D, rec.nmv, rec.energy, bond_eig_resnorm(rec))
	else
		@printf(io, "%s%s bond = %2d-%2d .. D = %4d nmv = %4d E = % .10f resnorm = %.2e (tol=%.0e)\n",
			prefix, arrow, b1, b2, D, rec.nmv, rec.energy, bond_eig_resnorm(rec), tol)
	end
	return rec
end

function print_bond_eig_records(records::Vector{BondEigRecord}; io::IO=stdout, prefix::String="", direction::String="", D::Int=0, tol::Union{Real,Nothing}=nothing)
	for rec in records
		print_bond_eig_record(rec; io=io, prefix=prefix, direction=direction, D=D, tol=tol)
	end
	return records
end

function print_bond_eig_sweep(st::DMRGSweepTiming; io::IO=stdout, prefix::String="", D::Int=0, tol::Union{Real,Nothing}=nothing)
	print_bond_eig_records(st.forward.bond_records; io=io, prefix=prefix, direction="forward", D=D, tol=tol)
	print_bond_eig_records(st.backward.bond_records; io=io, prefix=prefix, direction="backward", D=D, tol=tol)
	return st
end
