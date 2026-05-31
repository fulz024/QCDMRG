# renormalizeright2 is an inefficient version used for debug
function compute_workspace(mpsj::AbstractTensorMap)
    nl = 0
    nr = 0
    for (c, b) in blocks(mpsj)
        nl = max(nl, size(b, 1))
        nr = max(nr, size(b, 2))
    end
    return nl * nr
end

function mul_workspace!(workspace::Vector, a::AbstractMatrix, b::AbstractMatrix, α=true)
    s1, s3 = size(a, 1), size(b, 2)
    tmp = sreshape(StridedView(view(workspace, 1:s1*s3)), s1, s3)
    return mul!(tmp, a, b, α, false)
end

# y += a * x * b

function mul_twosides!(y::AbstractMatrix, a::AbstractMatrix, x::AbstractMatrix, b::AbstractMatrix, α, β, workspace::Vector)
    @assert (size(a, 1) == size(y, 1)) &&  (size(a, 2) == size(x, 1)) && (size(x, 2) == size(b, 1)) && (size(b, 2) == size(y, 2))
    # @assert Strided.isblasmatrix(y)
    # @assert Strided.isblasmatrix(a)
    # @assert Strided.isblasmatrix(x)
    # @assert Strided.isblasmatrix(b)
    m, n, k, l = size(a, 1), size(a, 2), size(x, 2), size(b, 2)
    cost1 = m * n * k + m * k * l
    cost2 = n * k * l + m * n * l
    if cost1 >= cost2
        tmp = mul_workspace!(workspace, x, b, α)
        mul!(y, a, tmp, true, β)
    else
        tmp = mul_workspace!(workspace, a, x, α)
        mul!(y, tmp, b, true, β) 
    end
    return y
end

# update based on renormalized operator
function updaterenormalizeright(hold::RBTensor, mpsA::MPSSiteTensor, mpsB::MPSSiteTensor, workspace::Vector=Vector{scalartype(hold)}(undef, compute_workspace(mpsA)))
    hnew = TensorMap(zeros, scalartype(hold), space_l(mpsB), space_l(mpsA))
    return updaterenormalizeright!(hnew, hold, mpsA, mpsB, workspace)
end
function updaterenormalizeright!(hnew::MPSBondTensor, hold::RBTensor, mpsA::MPSSiteTensor, mpsB::MPSTensor, workspace::Vector)
    @tensor hnew[1;6] += mpsB[1,2,3] * hold[2,3,4,5] * conj(mpsA[6,4,5])
    return hnew
end
function updaterenormalizeright!(hnew::MPSBondTensor, hold::RBTensor, mpsA::MPSTensor12, mpsB::MPSTensor12, workspace::Vector)
    for (f1l, f1r) in fusiontrees(hold)
        f2l = FusionTree((f1l.coupled,), f1l.coupled, (false,))
        mbj = get(mpsB, (f2l, f1l), nothing)
        if !isnothing(mbj)
            mb = StridedView(dropdims(mbj, dims=2))
            hl = StridedView(dropdims(hold[f1l, f1r], dims=(1,3)))
            ma = StridedView(dropdims(mpsA[f2l, f1r], dims=2))
            hr = hnew[f2l, f2l]
            # hr .+= mb * hl * ma'
            # tmp = mul_workspace!(workspace, mb, hl)
            # mul!(hr, tmp, ma', true, true)
            mul_twosides!(hr, mb, hl, ma', true, true, workspace)
        end
    end
    return hnew
end
function updaterenormalizeright(hold::MPSBondTensor, mpsA::MPSBondTensor, mpsB::MPSBondTensor, workspace::Vector=Vector{scalartype(hold)}(undef, compute_workspace(mpsA)))
    hnew = TensorMap(zeros, scalartype(hold), space_l(mpsB), space_l(mpsA))
    return updaterenormalizeright!(hnew, hold, mpsA, mpsB, workspace)
end
function updaterenormalizeright!(hnew::MPSBondTensor, hold::MPSBondTensor, mpsA::MPSBondTensor, mpsB::MPSBondTensor, workspace::Vector)
    @tensor hnew[1,4] += mpsB[1,2] * hold[2,3] * conj(mpsA[4,3])
    return hnew
end
function updaterenormalizeright(hold::RATensor, mpsA::MPSSiteTensor, mpsB::MPSSiteTensor, workspace::Vector=Vector{scalartype(hold)}(undef, compute_workspace(mpsA)))
    hnew = TensorMap(zeros, scalartype(hold), space_l(mpsB) ⊗ space(hold, 3) , space_l(mpsA))
    return updaterenormalizeright!(hnew, hold, mpsA, mpsB, workspace)    
end
function updaterenormalizeright!(hnew::MPSTensor, hold::RATensor, mpsA::MPSSiteTensor, mpsB::MPSTensor, workspace::Vector)
    @tensor hnew[1,4;7] += mpsB[1,2,3] * hold[2,3,4,5,6] * conj(mpsA[7,5,6])
    return hnew
end
function updaterenormalizeright!(hnew::MPSTensor, hold::RATensor, mpsA::MPSTensor12, mpsB::MPSTensor12, workspace::Vector)
    for (f1l, f1r) in fusiontrees(hold)
        f2r = FusionTree((f1l.uncoupled[1], f1l.uncoupled[2]), first(f1l.uncoupled[1] ⊗ f1l.uncoupled[2]), (f1l.isdual[1], f1l.isdual[2]))
        f2l = FusionTree((f2r.coupled,), f2r.coupled, (false,))
        mbj = get(mpsB, (f2l, f2r), nothing)
        if !isnothing(mbj)
            f3r = FusionTree((f1r.coupled,), f1r.coupled, (false,))
            maj = get(mpsA, (f3r, f1r), nothing)
            if !isnothing(maj)
                ma = StridedView(dropdims(maj, dims=2))
                hl = StridedView(dropdims(hold[f1l, f1r], dims=(1,3,4)))
                mb = StridedView(dropdims(mbj, dims=2))
                f3l = FusionTree((f2l.coupled, f1l.uncoupled[3]), first(f2l.coupled ⊗ f1l.uncoupled[3]), (false, false))
                hr = StridedView(dropdims(hnew[f3l, f3r], dims=2))
                # hr .+= mb * hl * ma'
                # tmp = mul_workspace!(workspace, mb, hl)
                # mul!(hr, tmp, ma', true, true)
                mul_twosides!(hr, mb, hl, ma', true, true, workspace)
            end
        end
    end
    return hnew
end
function updaterenormalizeright(hold::MPSTensor, mpsA::MPSBondTensor, mpsB::MPSBondTensor, workspace::Vector=Vector{scalartype(hold)}(undef, compute_workspace(mpsA)))
    hnew = TensorMap(zeros, scalartype(hold), space_l(mpsB) ⊗ space(hold, 2) , space_l(mpsA))
    return updaterenormalizeright!(hnew, hold, mpsA, mpsB, workspace)    
end
function updaterenormalizeright!(hnew::MPSTensor, hold::MPSTensor, mpsA::MPSBondTensor, mpsB::MPSBondTensor, workspace::Vector)
    @tensor hnew[1,3,5] += mpsB[1,2] * hold[2,3,4] * conj(mpsA[5,4])
    return hnew
end

function renormalizeright(hold::MPSBondTensor, mpoj::MPSBondTensor; kwargs...)
    cod1, cod2 = codomain(hold), codomain(mpoj)
    dom1, dom2 = domain(hold), domain(mpoj)
    cod = dom2' ⊗ cod1
    dom = cod2' ⊗ dom1	
    hnew = TensorMap(zeros, promote_type(scalartype(hold), scalartype(mpoj)), cod, dom)
    return renormalizeright!(hnew, hold, mpoj; kwargs...)
end
"""
    renormalizeright2!(hnew::RBTensor, hold::MPSBondTensor, mpoj::MPSBondTensor; add_adjoint::Bool=false)
The old inefficient version used for debug
"""
function renormalizeright2!(hnew::RBTensor, hold::MPSBondTensor, mpoj::MPSBondTensor; add_adjoint::Bool=false) 
    cod1, cod2 = codomain(hold), codomain(mpoj)
    dom1, dom2 = domain(hold), domain(mpoj)
    ((codomain(hnew) == dom2' ⊗ cod1) && (domain(hnew) == cod2' ⊗ dom1)) || throw(SpaceMismatch())
    if add_adjoint
        @tensor tmp[2,3;1,4] := mpoj[1,2] * hold[3,4]
        axpy!(true, tmp, hnew)
        axpy!(true, tmp', hnew)
    else
        @tensor hnew[2,3;1,4] += mpoj[1,2] * hold[3,4]
    end
	return hnew
end
function renormalizeright!(hnew::RBTensor, hold::MPSBondTensor, mpoj::MPSBondTensor; add_adjoint::Bool=false) 
    cod1, cod2 = codomain(hold), codomain(mpoj)
    dom1, dom2 = domain(hold), domain(mpoj)
    ((codomain(hnew) == dom2' ⊗ cod1) && (domain(hnew) == cod2' ⊗ dom1)) || throw(SpaceMismatch())

    # mpoj′ = permute(mpoj, (2,), (1,))
    mpoj′ = transpose(mpoj)
    for (f1l, f1r) in fusiontrees(mpoj′)
        c1 = f1l.coupled
        α = only(mpoj′[f1l, f1r])
        for (f2l, f2r) in fusiontrees(hold)
            c2 = f2l.coupled
            v = hold[f2l, f2r]
            c = first(c1 ⊗ c2)
            for (fl, coef1) in TK.merge(f1l, f2l, c)
                for (fr, coef2) in TK.merge(f1r, f2r, c)
                    out = sreshape(hnew[fl, fr], size(v))
                    coef = α * coef1 * coef2
                    # out .+= coef .* v
                    axpy!(coef, v, out)
                    if add_adjoint
                        # out .+= coef .* v'
                        axpy!(coef, v', out)
                    end
                end
            end
        end
    end
    return hnew
end
function renormalizeright(hold::MPSTensor, mpoj::MPSTensor; dagger::Bool=false)
    if dagger
        out = fuse(space(mpoj, 2), space(hold, 2)')
        hnew = scratch_rtensor!(scalartype(hold), space(mpoj, 3) ⊗ space(hold, 3)' ⊗ out, space(mpoj, 1)' ⊗ space(hold, 1)')
    else
        out = fuse(space(mpoj, 2), space(hold, 2))
        hnew = scratch_rtensor!(scalartype(hold), space(mpoj, 3) ⊗ space(hold, 1) ⊗ out, space(mpoj, 1)' ⊗ space(hold, 3)')
    end
    return renormalizeright!(hnew, hold, mpoj, dagger=dagger)
end
function renormalizeright2!(hnew::RATensor, hold::MPSTensor, mpoj::MPSTensor; dagger::Bool=false) 
    if dagger
        ((space(hnew, 1) == space(mpoj, 3)) && (space(hnew, 2) == space(hold, 3)') && 
            (space(hnew, 4) == space(mpoj, 1)) && (space(hnew, 5) == space(hold, 1)')) || throw(SpaceMismatch())
        (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

       tmp = DMRG.loose_isometry(storagetype(hnew), space(hnew, 3), space(mpoj, 2) ⊗ space(hold, 2)')
       @tensor hnew[3,6,7;1,4] += mpoj[1,2,3] * conj(hold[4,5,6]) * tmp[7,2,5]
    else
        ((space(hnew, 1) == space(mpoj, 3)) && (space(hnew, 2) == space(hold, 1)) && 
            (space(hnew, 4) == space(mpoj, 1)) && (space(hnew, 5) == space(hold, 3))) || throw(SpaceMismatch())
        (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

        tmp = DMRG.loose_isometry(storagetype(hnew), space(hnew, 3), space(mpoj, 2) ⊗ space(hold, 2))
        @tensor hnew[3,4,7;1,6] += mpoj[1,2,3] * hold[4,5,6] * tmp[7,2,5]
    end
    return hnew
end
function renormalizeright!(hnew::RATensor, hold::MPSTensor, mpoj::MPSTensor; dagger::Bool=false) 
    if dagger
        ((space(hnew, 1) == space(mpoj, 3)) && (space(hnew, 2) == space(hold, 3)') && 
            (space(hnew, 4) == space(mpoj, 1)) && (space(hnew, 5) == space(hold, 1)')) || throw(SpaceMismatch())
        (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

        (space(hnew, 3) ≅ space(mpoj, 2) ⊗ space(hold, 2)') || return hnew

        hold′ = hold'
        mpoj′ = permute(mpoj, (3,2), (1,))
        for (f1l, f1r) in fusiontrees(mpoj′)
            α = only(mpoj′[f1l, f1r])
            c1 = f1l.coupled
            for (f2l, f2r) in fusiontrees(hold′)
                v = StridedView(dropdims(hold′[f2l, f2r], dims=3))
                # println("f2l=", f2l, " f2r=", f2r)
                (f2l′, f2r′), coef0 = only(permute(f2l, f2r, (1,3), (2,)))
                # println("f2l′=", f2l′, " f2r′=", f2r′)
                c2 = f2l′.coupled
                c = first(c1 ⊗ c2)
                for (fl, coef1) in TK.merge(f1l, f2l′, c)
                    uncoupled = (fl.uncoupled[1], fl.uncoupled[3], first(fl.uncoupled[2] ⊗ fl.uncoupled[4]) )
                    # @assert fl.isdual[2] == fl.isdual[4]
                    isdual = (fl.isdual[1], fl.isdual[3], false)
                    fl′ = FusionTree(uncoupled, fl.coupled, isdual)
                    for (fr, coef2) in TK.merge(f1r, f2r′, c)
                        coef = α * coef0 * coef1 * coef2
                        out = sreshape(hnew[fl′, fr], size(v))
                        # out .+= coef .* v
                        axpy!(coef, v, out)
                    end
                end
            end
        end 
        return hnew
    else
        ((space(hnew, 1) == space(mpoj, 3)) && (space(hnew, 2) == space(hold, 1)) && 
            (space(hnew, 4) == space(mpoj, 1)) && (space(hnew, 5) == space(hold, 3))) || throw(SpaceMismatch())
        (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

        # (space(hnew, 3) ≅ space(mpoj, 2) ⊗ space(hold, 2)) || return hnew

        mpoj′ = permute(mpoj, (3,2), (1,))
        for (f1l, f1r) in fusiontrees(mpoj′)
            α = only(mpoj′[f1l, f1r])
            c1 = f1l.coupled
            for (f2l, f2r) in fusiontrees(hold)
                v = StridedView(dropdims(hold[f2l, f2r], dims=2))
                c2 = f2l.coupled
                c = first(c1 ⊗ c2)
                for (fl, coef1) in TK.merge(f1l, f2l, c)
                    uncoupled = (fl.uncoupled[1], fl.uncoupled[3], first(fl.uncoupled[2] ⊗ fl.uncoupled[4]) )
                    @assert fl.isdual[2] == fl.isdual[4]
                    isdual = (fl.isdual[1], fl.isdual[3], fl.isdual[2])
                    fl′ = FusionTree(uncoupled, fl.coupled, isdual)
                    for (fr, coef2) in TK.merge(f1r, f2r, c)
                        coef = α * coef1 * coef2
                        out = sreshape(hnew[fl′, fr], size(v))
                        # out .+= coef .* v
                        axpy!(coef, v, out)
                    end
                end
            end
        end
        return hnew
    end
end
function renormalizeright(hold::MPSTensor, mpoj1::Nothing)
    hnew = scratch_rtensor!(scalartype(hold), _u1u1_pspace' ⊗ space(hold, 1) ⊗ space(hold, 2), _u1u1_pspace' ⊗ space(hold, 3)')
    return renormalizeright!(hnew, hold, mpoj1)
end 
function renormalizeright2!(hnew::RATensor, hold::MPSTensor, mpoj1::Nothing) 
    mpoj = isomorphism(storagetype(hnew), _u1u1_pspace, _u1u1_pspace)
    (space(hnew, 3) == space(hold, 2)) || throw(SpaceMismatch())
    ((space(hnew, 1) == space(mpoj, 2)) && (space(hnew, 2) == space(hold, 1)) && 
        (space(hnew, 4) == space(mpoj, 1)) && (space(hnew, 5) == space(hold, 3))) || throw(SpaceMismatch())

    @tensor tmp[2,3,4;1,5] := mpoj[1,2] * hold[3,4,5]
    for (f1, f2) in fusiontrees(tmp)
        if fermionparity(f1.uncoupled[1]) && fermionparity(f1.uncoupled[3])
            lmul!(-1, tmp[f1, f2])
        end
    end
    axpy!(true, tmp, hnew)
    return hnew
end
function renormalizeright!(hnew::RATensor, hold::MPSTensor, mpoj1::Nothing) 
    (space(hnew, 3) == space(hold, 2)) || throw(SpaceMismatch())
    ((space(hnew, 1) == _u1u1_pspace') && (space(hnew, 2) == space(hold, 1)) && 
        (space(hnew, 4) == _u1u1_pspace) && (space(hnew, 5) == space(hold, 3))) || throw(SpaceMismatch())

    for (f2l, f2r) in fusiontrees(hold)
        v = StridedView(dropdims(hold[f2l, f2r], dims=2))
        c2 = f2l.coupled
        for c1 in sectors(_u1u1_pspace')
            c = first(c1 ⊗ c2)
            fl = FusionTree((c1, f2l.uncoupled[1], f2l.uncoupled[2]), c, (true, f2l.isdual[1], f2l.isdual[2]))
            fr = FusionTree((c1, f2r.uncoupled[1]), c, (true, f2r.isdual[1]))
            coef = (fermionparity(c1) && fermionparity(f2l.uncoupled[2])) ? -1 : 1
            out = sreshape(hnew[fl, fr], size(v))
            # out .+= coef .* v
            axpy!(coef, v, out)
        end
    end
    return hnew
end
function renormalizeright(hold::MPSBondTensor, mpoj::MPSTensor)
    T = promote_type(scalartype(hold), scalartype(mpoj))
    hnew = scratch_rtensor!(T, space(mpoj, 3) ⊗ space(hold, 1) ⊗ space(mpoj, 2), space(mpoj, 1)' ⊗ space(hold, 2)')
    return renormalizeright!(hnew, hold, mpoj)
end
function renormalizeright2!(hnew::RATensor, hold::MPSBondTensor, mpoj::MPSTensor) 
    (space(hnew, 3) == space(mpoj, 2)) || throw(SpaceMismatch())
    ((space(hnew, 1) == space(mpoj, 3)) && (space(hnew, 2) == space(hold, 1)) && 
        (space(hnew, 4) == space(mpoj, 1)) && (space(hnew, 5) == space(hold, 2))) || throw(SpaceMismatch())

    @tensor hnew[3,4,2;1,5] += mpoj[1,2,3] * hold[4,5] 
    return hnew
end
function renormalizeright!(hnew::RATensor, hold::MPSBondTensor, mpoj::MPSTensor) 
    (space(hnew, 3) == space(mpoj, 2)) || throw(SpaceMismatch())
    ((space(hnew, 1) == space(mpoj, 3)) && (space(hnew, 2) == space(hold, 1)) && 
        (space(hnew, 4) == space(mpoj, 1)) && (space(hnew, 5) == space(hold, 2))) || throw(SpaceMismatch())

    mpoj′ = permute(mpoj, (3,2), (1,))
    for (f1l, f1r) in fusiontrees(mpoj′)
        c1 = f1l.coupled
        α = only(mpoj′[f1l, f1r])
        for (f2l, f2r) in fusiontrees(hold)
            v = hold[f2l, f2r]
            c2 = f2l.coupled
            c = first(c1 ⊗ c2)
            for (fl, coef1) in TK.merge(f1l, f2l, c)
                fl′, coef0 = only(permute(fl, (1,3,2)))
                for (fr, coef2) in TK.merge(f1r, f2r, c)
                    coef = α * coef0 * coef1 * coef2
                    out = sreshape(hnew[fl′, fr], size(v))
                    # out .+= coef .* v
                    axpy!(coef, v, out)
                end
            end
        end
    end
    return hnew
end
function renormalizeright2!(hnew::RBTensor, hold::MPSTensor, mpoj::MPSTensor; add_adjoint::Bool=false) 
    (space(hold, 2) == space(mpoj, 2)') || throw(SpaceMismatch())
    ((space(hnew, 1) == space(mpoj, 3)) && (space(hnew, 2) == space(hold, 1)) && 
        (space(hnew, 3) == space(mpoj, 1)) && (space(hnew, 4) == space(hold, 3))) || throw(SpaceMismatch())
    if add_adjoint
        @tensor tmp[3,4;1,6] := mpoj[1,2,3] * hold[4,2,6] 
        axpy!(true, tmp, hnew)
        axpy!(true, tmp', hnew)
    else
        @tensor hnew[3,4;1,6] += mpoj[1,2,3] * hold[4,2,6] 
    end
    return hnew
end
function renormalizeright!(hnew::RBTensor, hold::MPSTensor, mpoj::MPSTensor; add_adjoint::Bool=false) 
    (space(hold, 2) == space(mpoj, 2)') || throw(SpaceMismatch())
    ((space(hnew, 1) == space(mpoj, 3)) && (space(hnew, 2) == space(hold, 1)) && 
        (space(hnew, 3) == space(mpoj, 1)) && (space(hnew, 4) == space(hold, 3))) || throw(SpaceMismatch())

    mpoj′ = permute(mpoj, (3,2), (1,))
    for (f1l, f1r) in fusiontrees(mpoj′)
        c1 = f1l.coupled
        α = only(mpoj′[f1l, f1r])
        for (f2l, f2r) in fusiontrees(hold)
            v = StridedView(dropdims(hold[f2l, f2r], dims=2))
            c2 = f2l.coupled
            c = first(c1 ⊗ c2)
            for (fl, coef1) in TK.merge(f1l, f2l, c)
                if fl.uncoupled[2] == conj(fl.uncoupled[4])
                    fl′ = FusionTree((fl.uncoupled[1], fl.uncoupled[3]), fl.coupled, (fl.isdual[1], fl.isdual[3]))
                    for (fr, coef2) in TK.merge(f1r, f2r, c)
                        coef = α * coef1 * coef2
                        out = sreshape(hnew[fl′, fr], size(v))
                        # out .+= coef .* v
                        axpy!(coef, v, out)
                        if add_adjoint
                            v′ = v'
                            out2 = sreshape(hnew[fr, fl′], size(v′))
                            # out2 .+= coef .* v′
                            axpy!(coef, v′, out2)
                        end
                    end 
                end
            end
        end
    end

    return hnew
end

# left update
function updaterenormalizeleft(hold::RBTensor, mpsA::MPSTensor, mpsB::MPSTensor, workspace::Vector=Vector{scalartype(hold)}(undef, compute_workspace(mpsA)))
    hnew = TensorMap(zeros, scalartype(hold), space_r(mpsA)', space_r(mpsB)')
    return updaterenormalizeleft!(hnew, hold, mpsA, mpsB, workspace)
end
function updaterenormalizeleft2!(hnew::MPSBondTensor, hold::RBTensor, mpsA::MPSTensor, mpsB::MPSTensor)
    @tensor hnew[3;6] += conj(mpsA[1,2,3]) * hold[1,2,4,5] * mpsB[4,5,6]
    return hnew
end
function updaterenormalizeleft!(hnew::MPSBondTensor, hold::RBTensor, mpsA::MPSTensor, mpsB::MPSTensor, workspace::Vector)
    for (f1l, f1r) in fusiontrees(hold)
        hl = StridedView(dropdims(hold[f1l, f1r], dims=(2,4)))
        f2r = FusionTree((f1r.coupled,), f1r.coupled, (false,))
        maj = get(mpsA, (f1l, f2r), nothing)
        if !isnothing(maj)
            ma = StridedView(dropdims(maj, dims=2))
            mb = StridedView(dropdims(mpsB[f1r, f2r], dims=2))
            hr = hnew[f2r, f2r]   
            # hr .+= ma' * hl * mb
            # tmp = mul_workspace!(workspace, ma', hl) 
            # mul!(hr, tmp, mb, true, true)
            mul_twosides!(hr, ma', hl, mb, true, true, workspace)
        end
    end
    return hnew
end
function updaterenormalizeleft(hold::MPSBondTensor, mpsA::MPSBondTensor, mpsB::MPSBondTensor, workspace::Vector=Vector{scalartype(hold)}(undef, compute_workspace(mpsA)))
    hnew = TensorMap(zeros, scalartype(hold), space_r(mpsA)', space_r(mpsB)')
    return updaterenormalizeleft!(hnew, hold, mpsA, mpsB, workspace)
end
function updaterenormalizeleft!(hnew::MPSBondTensor, hold::MPSBondTensor, mpsA::MPSBondTensor, mpsB::MPSBondTensor, workspace::Vector)
    @tensor hnew[2,4] += conj(mpsA[1,2]) * hold[1,3] * mpsB[3,4] 
    return hnew
end
function updaterenormalizeleft(hold::RATensor, mpsA::MPSTensor, mpsB::MPSTensor, workspace::Vector=Vector{scalartype(hold)}(undef, compute_workspace(mpsA)))
    hnew = TensorMap(zeros, scalartype(hold), space_r(mpsA)' ⊗ space(hold, 3), space_r(mpsB)')
    return updaterenormalizeleft!(hnew, hold, mpsA, mpsB, workspace)
end
function updaterenormalizeleft2!(hnew::MPSTensor, hold::RATensor, mpsA::MPSTensor, mpsB::MPSTensor)
    @tensor hnew[3,4;7] += conj(mpsA[1,2,3]) * hold[1,2,4,5,6] * mpsB[5,6,7]
    return hnew
end
function updaterenormalizeleft!(hnew::MPSTensor, hold::RATensor, mpsA::MPSTensor, mpsB::MPSTensor, workspace::Vector)
    for (f1l, f1r) in fusiontrees(hold)
        f2l = FusionTree((f1l.uncoupled[1], f1l.uncoupled[2]), first(f1l.uncoupled[1] ⊗ f1l.uncoupled[2]), (false, false))
        f2r = FusionTree((f2l.coupled,), f2l.coupled, (false,))
        maj = get(mpsA, (f2l, f2r), nothing)
        if !isnothing(maj)
            ma = StridedView(dropdims(maj, dims=2))
            f3r = FusionTree((f1r.coupled,), f1r.coupled, (false,))
            mbj = get(mpsB, (f1r, f3r), nothing)
            if !isnothing(mbj)
                mb = StridedView(dropdims(mbj, dims=2))
                hl = StridedView(dropdims(hold[f1l, f1r], dims=(2,3,5)))
                # ma = StridedView(dropdims(mpsA[f2l, f2r], dims=2))
                f3l = FusionTree((f2r.coupled, f1l.uncoupled[3]), f3r.coupled, (false, f1l.isdual[3]))
                hr = StridedView(dropdims(hnew[f3l, f3r], dims=2))
                # hr .+= ma' * hl * mb 
                # tmp = mul_workspace!(workspace, ma', hl)
                # mul!(hr, tmp, mb, true, true)
                mul_twosides!(hr, ma', hl, mb, true, true, workspace)
            end

        end
    end
    return hnew
end
function updaterenormalizeleft(hold::MPSTensor, mpsA::MPSBondTensor, mpsB::MPSBondTensor, workspace::Vector=Vector{scalartype(hold)}(undef, compute_workspace(mpsA)))
    hnew = TensorMap(zeros, scalartype(hold), space_r(mpsA)' ⊗ space(hold, 2), space_r(mpsB)')
    return updaterenormalizeleft!(hnew, hold, mpsA, mpsB, workspace)
end
function updaterenormalizeleft!(hnew::MPSTensor, hold::MPSTensor, mpsA::MPSBondTensor, mpsB::MPSBondTensor, workspace::Vector)
    @tensor hnew[2,3,5] += conj(mpsA[1,2]) * hold[1,3,4] * mpsB[4,5]
    return hnew
end

function renormalizeleft(hold::MPSBondTensor, mpoj::MPSBondTensor; kwargs...)
     hnew = TensorMap(zeros, promote_type(scalartype(hold), scalartype(mpoj)), space(hold, 1) ⊗ space(mpoj, 1), space(hold, 2)' ⊗ space(mpoj, 2)')
     return renormalizeleft!(hnew, hold, mpoj; kwargs...)
end
function renormalizeleft2!(hnew::RBTensor, hold::MPSBondTensor, mpoj::MPSBondTensor; add_adjoint::Bool=false) 
    (space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
        (space(hnew, 3) == space(hold, 2)) && (space(hnew, 4) == space(mpoj, 2)) || throw(SpaceMismatch())

    if add_adjoint
        @tensor tmp[1,3;2,4] := hold[1,2] * mpoj[3,4]
        axpy!(true, tmp, hnew)
        axpy!(true, tmp', hnew)
    else
        @tensor hnew[1,3;2,4] += hold[1,2] * mpoj[3,4]
    end
    return hnew
end
function renormalizeleft!(hnew::RBTensor, hold::MPSBondTensor, mpoj::MPSBondTensor; add_adjoint::Bool=false) 
    (space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
        (space(hnew, 3) == space(hold, 2)) && (space(hnew, 4) == space(mpoj, 2)) || throw(SpaceMismatch())

    for (f1l, f1r) in fusiontrees(hold)
        v = hold[f1l, f1r]
        c1 = f1l.coupled
        for (f2l, f2r) in fusiontrees(mpoj)
            α = only(mpoj[f2l, f2r])
            c2 = f2l.coupled
            c = first(c1 ⊗ c2)
            for (fl, coef1) in TK.merge(f1l, f2l, c)
                for (fr, coef2) in TK.merge(f1r, f2r, c)
                    coef = α * coef1 * coef2
                    out = sreshape(hnew[fl, fr], size(v))
                    # out .+= coef .* v
                    axpy!(coef, v, out)
                    if add_adjoint
                        # out .+= coef .* v'
                        axpy!(coef, v', out)
                    end
                end
            end
        end
    end
    return hnew
end
function renormalizeleft(hold::MPSTensor, mpoj::MPSTensor; dagger::Bool=false)
    T = promote_type(scalartype(hold), scalartype(mpoj))
    if dagger
        mspace = fuse(space(mpoj, 2)', space(hold, 2))       
        hnew = scratch_rtensor!(T, space(hold, 3)' ⊗ space(mpoj, 1) ⊗ mspace', space(hold, 1) ⊗ space(mpoj, 3)')
    else
        mspace = fuse(space(mpoj, 2)', space(hold, 2)')       
        hnew = scratch_rtensor!(T, space(hold, 1) ⊗ space(mpoj, 1) ⊗ mspace', space(hold, 3)' ⊗ space(mpoj, 3)')
    end
    return renormalizeleft!(hnew, hold, mpoj, dagger=dagger)
end
function renormalizeleft2!(hnew::RATensor, hold::MPSTensor, mpoj::MPSTensor; dagger::Bool=false) 
    if dagger
        (space(hnew, 1) == space(hold, 3)') && (space(hnew, 2) == space(mpoj, 1)) && 
            (space(hnew, 4) == space(hold, 1)') && (space(hnew, 5) == space(mpoj, 3)) || throw(SpaceMismatch())
        (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

        tmp = DMRG.loose_isometry(storagetype(hnew), space(hnew, 3), space(hold, 2)' ⊗ space(mpoj, 2) )
        @tensor hnew[3,4,7;1,6] += conj(hold[1,2,3]) * mpoj[4,5,6] * tmp[7,2,5]
    else
        (space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
            (space(hnew, 4) == space(hold, 3)) && (space(hnew, 5) == space(mpoj, 3)) || throw(SpaceMismatch())
        (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

        tmp = DMRG.loose_isometry(storagetype(hnew), space(hnew, 3), space(hold, 2) ⊗ space(mpoj, 2) )
        @tensor hnew[1,4,7;3,6] += hold[1,2,3] * mpoj[4,5,6] * tmp[7,2,5]
    end
    return hnew 
end
function renormalizeleft!(hnew::RATensor, hold::MPSTensor, mpoj::MPSTensor; dagger::Bool=false) 
    if dagger
        (space(hnew, 1) == space(hold, 3)') && (space(hnew, 2) == space(mpoj, 1)) && 
            (space(hnew, 4) == space(hold, 1)') && (space(hnew, 5) == space(mpoj, 3)) || throw(SpaceMismatch())
        (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

        (space(hnew, 3) ≅ space(hold, 2)' ⊗ space(mpoj, 2)) || return hnew

        hold′ = hold'
        for (f1l, f1r) in fusiontrees(hold′)
            v = StridedView(dropdims(hold′[f1l, f1r], dims=3))
            (f1l′, f1r′), coef0 = only(permute(f1l, f1r, (1,3), (2,)))
            c1 = f1l′.coupled
            for (f2l, f2r) in fusiontrees(mpoj)
                α = only(mpoj[f2l, f2r])
                c2 = f2l.coupled
                c = first(c1 ⊗ c2)
                for (fl, coef1) in TK.merge(f1l′, f2l, c)
                    uncoupled = (fl.uncoupled[1], fl.uncoupled[3], first(fl.uncoupled[2] ⊗ fl.uncoupled[4]) )
                    isdual = (fl.isdual[1], fl.isdual[3], true)
                    fl′ = FusionTree(uncoupled, fl.coupled, isdual)
                    for (fr, coef2) in TK.merge(f1r′, f2r, c)
                        coef = α * coef0 * coef1 * coef2
                        out = sreshape(hnew[fl′, fr], size(v))
                        # out .+= coef .* v
                        axpy!(coef, v, out)
                    end
                end
            end
        end      
    else
        (space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
            (space(hnew, 4) == space(hold, 3)) && (space(hnew, 5) == space(mpoj, 3)) || throw(SpaceMismatch())
        (dim(space(hnew, 3)) == dim(space(hold, 2)) == dim(space(mpoj, 2)) == 1) || throw(ArgumentError("middle space should be singlet"))

        for (f1l, f1r) in fusiontrees(hold)
            v = StridedView(dropdims(hold[f1l, f1r], dims=2))
            c1 = f1l.coupled
            for (f2l, f2r) in fusiontrees(mpoj)
                α = only(mpoj[f2l, f2r])
                c2 = f2l.coupled
                c = first(c1 ⊗ c2)
                for (fl, coef1) in TK.merge(f1l, f2l, c)
                    uncoupled = (fl.uncoupled[1], fl.uncoupled[3], first(fl.uncoupled[2] ⊗ fl.uncoupled[4]) )
                    isdual = (fl.isdual[1], fl.isdual[3], true)
                    fl′ = FusionTree(uncoupled, fl.coupled, isdual)
                    for (fr, coef2) in TK.merge(f1r, f2r, c)
                        coef = α * coef1 * coef2
                        out = sreshape(hnew[fl′, fr], size(v))
                        # out .+= coef .* v
                        axpy!(coef, v, out)
                    end                   
                end
            end
        end
    end
    return hnew 
end
function renormalizeleft(hold::MPSTensor, mpoj1::Nothing)
    hnew = scratch_rtensor!(scalartype(hold), space(hold, 1) ⊗ _u1u1_pspace ⊗ space(hold, 2), space(hold, 3)' ⊗ _u1u1_pspace)
    return renormalizeleft!(hnew, hold, mpoj1)
end
function renormalizeleft2!(hnew::RATensor, hold::MPSTensor, mpoj1::Nothing)
    mpoj = isomorphism(storagetype(hnew), _u1u1_pspace, _u1u1_pspace)
    (space(hnew, 3) == space(hold, 2)) || throw(SpaceMismatch())
    (space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
        (space(hnew, 4) == space(hold, 3)) && (space(hnew, 5) == space(mpoj, 2)) || throw(SpaceMismatch())

    @tensor tmp[1,4,2;3,5] := hold[1,2,3] * mpoj[4,5] 
    for (f1, f2) in fusiontrees(tmp)
        if fermionparity(f1.uncoupled[3]) && fermionparity(f2.uncoupled[2])
            lmul!(-1, tmp[f1, f2])
        end
    end   
    axpy!(true, tmp, hnew)
    return hnew   
end
function renormalizeleft!(hnew::RATensor, hold::MPSTensor, mpoj1::Nothing)
    (space(hnew, 3) == space(hold, 2)) || throw(SpaceMismatch())
    (space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == _u1u1_pspace) && 
        (space(hnew, 4) == space(hold, 3)) && (space(hnew, 5) == _u1u1_pspace') || throw(SpaceMismatch())

    for (f1l, f1r) in fusiontrees(hold)
        v = StridedView(dropdims(hold[f1l, f1r], dims=2))
        c1 = f1l.coupled
        for c2 in sectors(_u1u1_pspace)
            c = first(c1 ⊗ c2)
            fl = FusionTree((f1l.uncoupled[1], c2, f1l.uncoupled[2]), c, (f1l.isdual[1], false, f1l.isdual[2]))
            fr = FusionTree((f1r.uncoupled[1], c2), c, (f1r.isdual[1], false))
            coef = (fermionparity(f1l.uncoupled[2]) && fermionparity(c2)) ? -1 : 1
            out = sreshape(hnew[fl, fr], size(v))
            # out .+= coef .* v
            axpy!(coef, v, out)
        end
    end
    return hnew   
end
function renormalizeleft(hold::MPSBondTensor, mpoj::MPSTensor)
    T = promote_type(scalartype(hold), scalartype(mpoj))
    hnew = scratch_rtensor!(T, space(hold, 1) ⊗ space(mpoj, 1) ⊗ space(mpoj, 2), space(hold, 2)' ⊗ space(mpoj, 3)')
    return renormalizeleft!(hnew, hold, mpoj)
end
function renormalizeleft2!(hnew::RATensor, hold::MPSBondTensor, mpoj::MPSTensor) 
    (space(hnew, 3) == space(mpoj, 2)) || throw(SpaceMismatch())
    (space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
        (space(hnew, 4) == space(hold, 2)) && (space(hnew, 5) == space(mpoj, 3)) || throw(SpaceMismatch())

    @tensor hnew[1,3,4;2,5] += hold[1,2] * mpoj[3,4,5] 
    return hnew
end
function renormalizeleft!(hnew::RATensor, hold::MPSBondTensor, mpoj::MPSTensor) 
    (space(hnew, 3) == space(mpoj, 2)) || throw(SpaceMismatch())
    (space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
        (space(hnew, 4) == space(hold, 2)) && (space(hnew, 5) == space(mpoj, 3)) || throw(SpaceMismatch())

    # @tensor hnew[1,3,4;2,5] += hold[1,2] * mpoj[3,4,5] 
    for (f1l, f1r) in fusiontrees(hold)
        v = hold[f1l, f1r]
        c1 = f1l.coupled
        for (f2l, f2r) in fusiontrees(mpoj)
            α = only(mpoj[f2l, f2r])
            c2 = f2l.coupled
            c = first(c1 ⊗ c2)
            for (fl, coef1) in TK.merge(f1l, f2l, c)
                for (fr, coef2) in TK.merge(f1r, f2r, c)
                    coef = α * coef1 * coef2
                    out = sreshape(hnew[fl, fr], size(v))
                    # out .+= coef .* v
                    axpy!(coef, v, out)
                end
            end
        end
    end
    return hnew
end
function renormalizeleft2!(hnew::RBTensor, hold::MPSTensor, mpoj::MPSTensor; add_adjoint::Bool=false) 
    (space(hold, 2) == space(mpoj, 2)') || throw(SpaceMismatch())
    ((space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
        (space(hnew, 3) == space(hold, 3)) && (space(hnew, 4) == space(mpoj, 3))) || throw(SpaceMismatch())
    if add_adjoint
        @tensor tmp[1,4;3,6] := hold[1,2,3] * mpoj[4,2,6] 
        axpy!(true, tmp, hnew)
        axpy!(true, tmp', hnew)
    else
        @tensor hnew[1,4;3,6] += hold[1,2,3] * mpoj[4,2,6] 
    end
    return hnew   
end
function renormalizeleft!(hnew::RBTensor, hold::MPSTensor, mpoj::MPSTensor; add_adjoint::Bool=false) 
    (space(hold, 2) == space(mpoj, 2)') || throw(SpaceMismatch())
    ((space(hnew, 1) == space(hold, 1)) && (space(hnew, 2) == space(mpoj, 1)) && 
        (space(hnew, 3) == space(hold, 3)) && (space(hnew, 4) == space(mpoj, 3))) || throw(SpaceMismatch())
    # if add_adjoint
    #     @tensor tmp[1,4;3,6] := hold[1,2,3] * mpoj[4,2,6] 
    #     axpy!(true, tmp, hnew)
    #     axpy!(true, tmp', hnew)
    # else
    #     @tensor hnew[1,4;3,6] += hold[1,2,3] * mpoj[4,2,6] 
    # end
    for (f1l, f1r) in fusiontrees(hold)
        v = StridedView(dropdims(hold[f1l, f1r], dims=2))
        c1 = f1l.coupled
        for (f2l, f2r) in fusiontrees(mpoj)
            α = only(mpoj[f2l, f2r])
            c2 = f2l.coupled
            c = first(c1 ⊗ c2)
            for (fl, coef1) in TK.merge(f1l, f2l, c)
                if fl.uncoupled[2] == conj(fl.uncoupled[4])
                    fl′ = FusionTree((fl.uncoupled[1], fl.uncoupled[3]), c, (fl.isdual[1], fl.isdual[3]))
                    for (fr, coef2) in TK.merge(f1r, f2r, c)
                        coef = α * coef1 * coef2
                        out = sreshape(hnew[fl′, fr], size(v))
                        # out .+= coef .* v
                        axpy!(coef, v, out)
                        if add_adjoint
                            v′ = v'
                            out2 = sreshape(hnew[fr, fl′], size(v′))
                            # out2 .+= coef .* v′
                            axpy!(coef, v′, out2)
                        end
                    end
                end
            end
        end
    end
    return hnew   
end

