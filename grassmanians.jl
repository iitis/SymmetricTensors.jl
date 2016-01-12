# module   Grassmanians
type Grass
  dim::Tuple{Int,Int}     # dimensions
  data::Matrix{Float64}     # point on manifold, orthogonal matrix of size dim
  base     # base for the tangent space at the current point
  proj     # projection on tangent spaces in matrix form
  tan      # tangent vector of movement
  tan2     # tangent vector (arbitrary) to be transported
  rtan     # local coordinates of tan in tangent space basis
  rtan2    # local coordinates of tan2 in ---- || ------
  svd::Base.LinAlg.SVD   # SVD of tan, used for geodesic movement
  rsvd::Base.LinAlg.SVD     # SVD of the local coordinates of rtan

  Grass()=new()
end

function Grass(dim::Array{Int,1})
  if size(dim) == (2,)
    g = Grass()
    dump((dim...))
    g.dim = (dim...)
    return g
  end
end

# SET  -  Set specified properties of a grass object.
#
# Properties are: dim, data, base, proj,
#                 tan, tan2, rtan, rtan2,
#                 svd, rsvd
#
# Examples, g grass object, D orthonormal matrix, T tangent vector
#   g = set(g, 'data', D)
#   g = set(g, 'data', D, 'tan', T)

function set!(g::Grass,prop::Symbol,val::Any)
  if prop == :dim
     g.dim = (val...)
  elseif prop ==  :data
     g.data = val
    #  if isdefined(g.dim)
         g.dim = (size(val)...)
    #  end
  elseif prop == :base
     Q,r = qr(val, thin=false)
    #  g.base = Q[:,g.dim[2]+1:end]
    g.base = Q[:,g.dim[2]+1:end]
  elseif prop ==  :proj
     g.proj = eye(size(val,1)) - val*val'
  elseif prop ==  :tan
     g.tan = val
     g.svd = svdfact(val)
  elseif prop ==  :tan2
     g.tan2 = val
  elseif prop ==  :rtan
     g.rtan = val
     g.rsvd = svdfact(val, thin=true)
  elseif prop ==  :rtan2
     g.rtan2 = val
  elseif prop ==  :svd
     g.svd = val
  elseif prop ==  :rsvd
     g.rsvd = val
  else
     error("Invalid property of grass object.")
  end
end


# NORM  -  Compute norm of a tangent vector for a grass object.
#
# Examples, g grass object, prop is one of tan, rtan, tan2 and rtan2.
#   n = norm(g,'tan')
#   n = norm(g,'rtan')
#
# Copyright 2008.
# Berkant Savas, LinkÃ¶ping University.


## tanVec is one of the following: tan, tan2, rtan, rtan2
    # In the metric on Grassmann manifolds the norm is
    # < tanVec , tanVec > = trace( tanVec' * tanVec )
function norm(g::Grass, tanVec::Symbol)
    if tanVec == :tan
        n = sum( sum( g.tan .* g.tan ) )
    elseif tanVec == :rtan
        n = sum( sum( g.rtan .* g.rtan ) )
    elseif tanVec == :tan2
        n = sum( sum( g.tan2 .* g.tan2 ) )
    elseif tanVec == :rtan2
        n = sum( sum( g.rtan2 .* g.rtan2 ) )
    end
    return n
end

# Inner product between tangent vectors g.tan and g.tan2
# defined on the tangent space for the point g.data.
#
# Example: g grass object, both tangent vectors initiated.
#   val = innerProd(g)
#
# Copyright 2008.
# Berkant Savas, LinkÃ¶ping University.
innerProd(g::Grass) = sum( sum( g.tan .* g.tan2 ) )


# MOVE  -  move the data point and/or
# parallel transport tangent vectos.
#
# Examples, g grass object, t is the step size, and prop is one of
#           'p'     - move only the data point,
#           'ptt'   - move the point and parallel transport both
#                     tangent vectors,
#           'pb'    - move the data point and update the basis.
#
#
#   g = move(g, t, 'p')
#   g = move(g, t, 'ptt')
#
# Copyright 2008.
# Berkant Savas, LinkÃ¶ping University.
function move(g::Grass,t::Float64,prop)
    X   = g.data
    if prop == :p
        s = g.svd
        X = movePoint(X,s,t)
        X = orthonormalize(X)
        g.data = X
    elseif prop == :ptt
        s   = g.svd
        X = movePoint(X,s,t)
        X = X * s.Vt'
        T = [g.data*s.Vt s.U] * [-diagm(sin(t*s.S)); diagm(cos(t*s.S))]
        X = orthonormalize(X)
        g.data = X
        g = moveTan(T,g,s)
        g = moveTan2(T,g,s)
    elseif prop == :pb
        B = g.base
        s = g.rsvd
        T1  = X*s.Vt
        T2  = B*s.U
        X = movePoint2(T1,T2,s,t)
        B = moveBase(B,T1,T2,s,t)
        X = orthonormalize(X)
        g.data = X
        g.base = B
    else
        error("Available properties are, p, ptt ,and pb.")
    end
    return g
end


# Move the data point g.data along geodesic given by g.tan
# without the V-matrix multiplied from the right, Edelman98:327
movePoint(X::Matrix, s::Base.LinAlg.SVD, t::Float64) = X*s.Vt*diagm(cos(t*s.S)) + s.U*diagm(sin(t*s.S))

# Move the data point g.data along geodesic given by g.tan
# with the V-matrix multiplied from the right. Keep the base in
# the process.
movePoint2(T1,T2,s::Base.LinAlg.SVD,t::Float64)  = (T1*diagm(cos(t*s.S)) + T2*diagm(sin(t*s.S)))*s.Vt'


# Move the base for the tangent plane.
moveBase(B,T1,T2,s::Base.LinAlg.SVD,t::Float64) = ( T1*diagm(-sin(t*s.S)) + T2*diagm(cos(t*s.S)))*s.U' +  B - T2*s.U'

# Move tangent vector g.tan along the geodesic given by g.tan.
# The SVD of g.tan at the new point is also computed and updated.
function moveTan(T,g::Grass,s::Base.LinAlg.SVD)
    g.tan = T * diagm(s.S) * s.Vt'    # g.tan = T * s.S * s.V' -- uwaga na transpozycję
    g.svd = svdfact(g.tan, thin=true)
    return g
end

# Move tangent vector g.tan2 along the geodesic given by g.tan
function moveTan2(T,g::Grass,s::Base.LinAlg.SVD)
    T2 = s.U'*g.tan2
    g.tan2  = T*T2 + g.tan2 - s.U*T2
    return g
end

orth(x)=svd(x)[1]

# Reorthogonalize if not on the manifold.
function orthonormalize(X::Matrix)
  if abs(LinAlg.norm(X'*X - eye(size(X,2)), 2)) > 1e-12
      X = orth(X)
  end
  X
end
# end

function subspace(A::Matrix, B::Matrix)
  A = orth(A)
  B = orth(B)
  c = A'*B
  scos = minimum(svd(c)[2])
  if (scos^2 > 1/2)
    if (size(A,2) >= size(B,2))
      c = B - A*c
    else
      c = A - B*c'
    end
    ssin = maximum(svd(c)[2])
    ang = asin(min(ssin, 1))
  else
    ang = acos(scos)
  end
  return ang
end
