const RBTensor{S} = AbstractTensorMap{S, 2, 2}
const MPSTensor12{S} = AbstractTensorMap{S, 1, 2}
const MPSSiteTensor{S} = Union{MPSTensor{S}, MPSTensor12{S}}
DMRG.space_l(t::AbstractTensorMap{S, 1, 2}) where {S} = space(t, 1)
DMRG.space_r(t::AbstractTensorMap{S, 1, 2}) where {S} = space(t, 3)


struct RenormalizedATensor{S<:ElementarySpace, M<:MPSTensor, F, I<:Sector}
	data::M
    codom::ProductSpace{S,3}
    dom::ProductSpace{S,2}
	outer::SectorDict{I,FusionTreeDict{F,UnitRange{Int}}}
end


function RenormalizedATensor(f, codom::ProductSpace{S, 3}, dom::ProductSpace{S, 2}) where {S}
	(FusionStyle(sectortype(S)) isa UniqueFusion) || throw(ArgumentError("only Abelian sector allowed"))
	((codom[1] == dom[1]) && (codom[2] == dom[2])) || throw(SpaceMismatch())
	codom2 = fuse(codom[1], codom[2]) ⊗ codom[3]
	dom2 = ⊗(fuse(dom[1], dom[2]))
	data = TensorMap(f, codom2, dom2)
	rowr, rowdims = TK._buildblockstructure(dom, blocksectors(dom2))
	return RenormalizedATensor(data, codom, dom, rowr)
end
RenormalizedATensor(f, ::Type{T}, codom::ProductSpace{S, 3}, dom::ProductSpace{S, 2}) where {T<:Number, S} = RenormalizedATensor(s->f(T, s), codom, dom)

TK.domain(t::RenormalizedATensor) = t.dom
TK.codomain(t::RenormalizedATensor) = t.codom
TK.scalartype(::Type{RenormalizedATensor{S, M, F, I}}) where {S, M, F, I} = scalartype(M)
TK.space(t::RenormalizedATensor) = HomSpace(codomain(t), domain(t))
TK.space(t::RenormalizedATensor, i::Int) = space(t)[i]

function Base.similar(t::RenormalizedATensor, ::Type{T}) where {T}
	data = similar(t.data, T)
	return RenormalizedATensor(data, t.codom, t.dom, t.outer)
end

Base.similar(t::RenormalizedATensor) = similar(t, scalartype(t))


rbtensortype(::Type{S}, ::Type{T}) where {S <: ElementarySpace, T} = tensormaptype(S, 2, 2, T)

# which one to use?
# ratensortype(::Type{S}, ::Type{T}) where {S <: ElementarySpace, T} = tensormaptype(S, 3, 2, T)
# const RATensor{S} = AbstractTensorMap{S, 3, 2}
# RATensor(f, codom::ProductSpace{S, 3}, dom::ProductSpace{S, 2}) where {S <: ElementarySpace} = TensorMap(f, codom, dom)
# RATensor(f, T::Type, codom::ProductSpace{S, 3}, dom::ProductSpace{S, 2}) where {S <: ElementarySpace} = TensorMap(f, T, codom, dom)


function ratensortype(::Type{S}, ::Type{T}) where {S <: ElementarySpace, T} 
	M = mpstensortype(S, T)
	I = sectortype(S)
	F = TK.fusiontreetype(I, 2)
	return RenormalizedATensor{S, M, F, I}
end
const RATensor{S} = RenormalizedATensor{S}
RATensor(f, codom::ProductSpace{S, 3}, dom::ProductSpace{S, 2}) where {S <: ElementarySpace} = RenormalizedATensor(f, codom, dom)
RATensor(f, T::Type, codom::ProductSpace{S, 3}, dom::ProductSpace{S, 2}) where {S <: ElementarySpace} = RenormalizedATensor(f, T, codom, dom)


@inline function Base.getindex(t::RenormalizedATensor{S, M, F, I}, f1::FusionTree{I, 3}, f2::FusionTree{I, 2}) where {S, M, F, I}
	c = f1.coupled
	(c == f2.coupled) || throw(SectorMismatch())
	inner_f1 = FusionTree((first(f1.uncoupled[1] ⊗ f1.uncoupled[2]), f1.uncoupled[3]), c, (false, f1.isdual[3]))
	inner_f2 = FusionTree((f2.coupled,), c, (false,))
	# t_inner = t.data
	# v = StridedView(t_inner.data[c])[t_inner.rowr[c][inner_f1], t_inner.colr[c][inner_f2]]
	v = t.data[inner_f1, inner_f2]
	outer_f1 = FusionTree((f1.uncoupled[1], f1.uncoupled[2]), inner_f1.uncoupled[1], (f1.isdual[1], f1.isdual[2]))
	r1 = t.outer[outer_f1.coupled][outer_f1]
	r2 = t.outer[inner_f2.coupled][f2]
	d = (dims(codomain(t), f1.uncoupled)..., dims(domain(t), f2.uncoupled)...)
	return sreshape(v[r1, :, r2], d)

end
@propagate_inbounds Base.setindex!(t::RenormalizedATensor{S, M, F, I}, v, f1::FusionTree{I, 3}, f2::FusionTree{I, 2}) where {S, M, F, I} = copy!(getindex(t, f1, f2), v)

struct RenormalizedBlockIterator{I<:Sector, F₁<:FusionTree{I, 2},F₂<:FusionTree{I, 1}}
	inner::TensorKeyIterator{I, F₁, F₂}
	outer::SectorDict{I,FusionTreeDict{F₁,UnitRange{Int}}}
end

Base.IteratorSize(::Type{<:RenormalizedBlockIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:RenormalizedBlockIterator}) = Base.HasEltype()
function Base.eltype(T::Type{<:RenormalizedBlockIterator{I}}) where {I}
	F₁ = TK.fusiontreetype(I, 3)
	F₂ = TK.fusiontreetype(I, 2)
	return Tuple{F₁,F₂}
end 
function Base.length(it::RenormalizedBlockIterator)
	l = 0
	inner, outer = it.inner, it.outer
    for (rowdict, coldict) in zip(values(inner.rowr), values(inner.colr))
    	for (f1, r1) in rowdict, (f2, r2) in coldict
    		l += length(outer[f1.uncoupled[1]]) * length(outer[f2.uncoupled[1]])
    	end
    end
    return l
end

function Base.iterate(it::RenormalizedBlockIterator)
	inner_next = iterate(it.inner)
	isnothing(inner_next) && return nothing
	(f₁, f₂), inner_state = inner_next
	rowdict, coldict = it.outer[f₁.uncoupled[1]], it.outer[f₂.uncoupled[1]]
	# rowdict and coldict should not be empty
	(fl, r1), rowstate = iterate(rowdict)
	(fr, r2), colstate = iterate(coldict)
	fl′ = FusionTree((fl.uncoupled..., f₁.uncoupled[2]), f₁.coupled, (fl.isdual..., f₁.isdual[2]))
	return (fl′, fr), (f₁, fr, inner_state, rowdict, rowstate, coldict, colstate)
end

function Base.iterate(it::RenormalizedBlockIterator, state)
	f₁, fr, inner_state, rowdict, rowstate, coldict, colstate = state
	# @assert f₁.coupled == fr.coupled
	rownext = iterate(rowdict, rowstate)
	if !isnothing(rownext)
		(fl, r1), rowstate = rownext
		fl′ = FusionTree((fl.uncoupled..., f₁.uncoupled[2]), f₁.coupled, (fl.isdual..., f₁.isdual[2]))
		# @assert fl′.coupled == fr.coupled
		return (fl′, fr), (f₁, fr, inner_state, rowdict, rowstate, coldict, colstate)
	end
	colnext = iterate(coldict, colstate)
	if !isnothing(colnext)
		rownext = iterate(rowdict)
		@assert rownext !== nothing
		(fl, r1), rowstate = rownext
		(fr′, r2), colstate = colnext
		fl′ = FusionTree((fl.uncoupled..., f₁.uncoupled[2]), f₁.coupled, (fl.isdual..., f₁.isdual[2]))
		@assert fl′.coupled == fr′.coupled
		return (fl′, fr′), (f₁, fr′, inner_state, rowdict, rowstate, coldict, colstate)
	end
	inner_next = iterate(it.inner, inner_state)
	isnothing(inner_next) && return nothing
	(f₁, f₂), inner_state = inner_next
	rowdict, coldict = it.outer[f₁.uncoupled[1]], it.outer[f₂.uncoupled[1]]
	(fl, r1), rowstate = iterate(rowdict)
	(fr′, r2), colstate = iterate(coldict)
	fl′ = FusionTree((fl.uncoupled..., f₁.uncoupled[2]), f₁.coupled, (fl.isdual..., f₁.isdual[2]))
	# @assert fl′.coupled == fr′.coupled
	return (fl′, fr′), (f₁, fr′, inner_state, rowdict, rowstate, coldict, colstate)		

end

TK.fusiontrees(t::RenormalizedATensor) = RenormalizedBlockIterator(fusiontrees(t.data), t.outer)

function Base.convert(::Type{<:TensorMap}, t::RenormalizedATensor)
	t2 = TensorMap(zeros, scalartype(t), t.codom, t.dom)
	for (fl, fr) in fusiontrees(t2)
		copy!(t2[fl, fr], t[fl, fr])
	end
	return t2
end

function Base.convert(::Type{<:RenormalizedATensor}, t::AbstractTensorMap{S, 3, 2}) where {S}
	t2 = RenormalizedATensor(zeros, scalartype(t), codomain(t), domain(t))
	for (fl, fr) in fusiontrees(t2)
		copy!(t2[fl, fr], t[fl, fr])
	end
	return t2
end

function renormalizedoperator(t::Union{RBTensor, MPSSiteTensor})
	codom2, dom2 = fuse(codomain(t)), fuse(domain(t))
	return TensorMap(blocks(t), codom2, dom2)
end
renormalizedoperator(t::RenormalizedATensor) = t.data
renormalizedoperator(t::AbstractTensorMap{S, 3, 2}) where {S} = renormalizedoperator(convert(RenormalizedATensor, t))
