using FactCheck
import Base.norm
include("../src/grassmanians.jl")

facts("test orth") do
  n, r = 2, 2
  X = orth(randn(n, r))
  @fact norm(X'*X - eye(r)) --> roughly(0., atol=1e-14)

  n, r = 42, 13
  X = orth(randn(n, r))
  @fact norm(X'*X - eye(r)) --> roughly(0., atol=1e-14)

  n, r = 5, 2
  X = orth(randn(n, r))
  Tg  = (eye(n) - X*X')*randn(n,r)
  Tg2 = (eye(n) - X*X')*randn(n,r)

  @fact norm(Tg'*X) --> roughly(0., atol=1e-14)
  @fact norm(Tg2'*X) --> roughly(0., atol=1e-14)
end

facts("test set") do

  context("test simple methods") do
    n, r = 5, 2
    X = orth(randn(n, r))
    g = Grass((n, r))
    set!(g, :data, X)
    @fact g.data --> X
    @fact g.dim --> (n, r)

    n, r = 5, 3
    X = orth(randn(n, r))
    set!(g, :data, X)
    @fact g.data --> X
    @fact g.dim --> (n, r)

    Tg  = (eye(n) - X*X')*randn(n,r)
    Tg2 = (eye(n) - X*X')*randn(n,r)

    set!(g, :tan, Tg)
    @fact g.tan --> Tg
    @fact g.svd --> svdfact(Tg)

    set!(g, :tan2, Tg2)
    @fact g.tan2 --> Tg2
  end
end

exit(42)
X  = orth(randn(n,r))  # A point on Gr(n,r)


# Global coordinates for two tangents.

# Declare the grass-object and initiate some fields.
g = Grass([n,r])
set!(g,:data,X)
set!(g,:tan,Tg)
set!(g,:tan2,Tg2)
set!(g,:base,X)

Xp = g.base
# Local coordinate for the two tangents.
Tl  = Xp'*Tg
println("TLLL")
Tl2 = Xp'*Tg2

## Tests and operations.
g.data |> println
# The svd of first tangent is already set!
s = g.svd
println(size(s.U), " ", size(diagm(s.S)), " ", size(s.Vt))

# Verify this is the case
LinAlg.norm( s.U*diagm(s.S)*s.Vt' - g.tan ) |> println

# Compute the norm of first and second tangent.
norm(g,:tan) |> println
norm(g,:tan2) |> println
#norm(g,'rtan2')         # rtan2-field not yet defined.
set!(g,:rtan2,Tl2)
LinAlg.norm(g.rtan2) |> println

# Compute the inner product between the first and second tangent.
innerProd(g) |> println

## Move operations

# t is the step length
t = 0.5

# Move just the point X.
g2 = move(g,t,:p)
X2 = g2.data
# X2 is a point on the manifold!
X2'*X2

# But tangents in g2 are not transported!
X2'*g2.tan
X2'*g2.tan2

# Nor the basis matrix is transproted.
X2'*g2.base

# Now move the point ant both tangents.
g3 = move(g,t,:ptt)
X3 = g3.data

# Now tangents are transported and of courese they
# are orthogonal to the current point.
X3'*g3.tan
X3'*g3.tan2

# Compute the inner product again, and compare with
# the computation at the previous point: innerProd(g)
innerProd(g3)

# Move the point and the basis matrix.

# First set the first tangent (direction of movement)
# in local coordinates. The rsvd-field is set automatically.
println("TLLL")
size(Tl) |>println
set!(g,:rtan,Tl)
g4 = move(g,t,:pb)
X4 = g4.data
X4'*g4.base |>println
# The matrices X2,X3,X4 represent the same point/subspce
subspace(X2,X3) |>println
subspace(X2,X4) |>println

# But X2 is different from X3 and X4.
X2-X3 |>println
