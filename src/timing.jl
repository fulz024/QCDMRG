using Printf

"""
Per half-sweep timings.

- `Teff`: assemble `QCCenter` (`renormalizedstorage` + `terms`; block2 effective-H build only)
- `Teig`: Davidson / Lanczos (includes all `H|ψ⟩` matvec)
- `Tmve`: environment shift (`renormalizestorage*`) + post-SVD `updatestoragerenormalize*` / `setstorage!`
- `Tsvd`: two-site SVD truncation
- `Tsplt`: split / merge MPS after SVD (normalize, bond tensors, energy check)
"""
mutable struct DMRGTiming
	teff::Float64
	teig::Float64
	tmve::Float64
	tsvd::Float64
	tsplt::Float64
	nbonds::Int
end

DMRGTiming() = DMRGTiming(0, 0, 0, 0, 0, 0)

function reset!(t::DMRGTiming)
	t.teff = t.teig = t.tmve = t.tsvd = t.tsplt = 0.0
	t.nbonds = 0
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
	if t.nbonds > 0
		@printf(io, " | nbonds = %d", t.nbonds)
	end
	println(io)
	return t
end

function print_dmrg_sweep_timing(st::DMRGSweepTiming; io::IO=stdout, prefix::String="")
	print_dmrg_timing(st.forward; io=io, prefix=prefix, direction="forward")
	print_dmrg_timing(st.backward; io=io, prefix=prefix, direction="backward")
	return st
end
