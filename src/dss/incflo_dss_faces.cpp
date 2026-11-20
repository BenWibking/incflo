// Generate local reverse derivatives of the native periodic MOL face operators.
// This translation unit can also emit scalar LLVM IR for a HIP device build.
#include <AMReX_BCRec.H>
#include <hydro_mol_edge_state_K.H>
using namespace amrex;

extern "C" {
extern int enzyme_dup, enzyme_const;
void __enzyme_autodiff(void*, ...);
}

namespace {
// Four consecutive cells straddle one face. On a periodic Cartesian grid the
// MOL reconstruction is the same in each direction, so use the native x kernel.
Array4<Real const> line(Real const* values, int count)
{
    return {values, IntVect(0), IntVect(count,1,1), 1};
}

Real fluxPair(Real const* q, Real left, Real right)
{
    Real const speed = q[4];
    Real const face_values[3] = {0, 0, speed};
    BCRec const periodic(BCType::int_dir, BCType::int_dir, BCType::int_dir,
                         BCType::int_dir, BCType::int_dir, BCType::int_dir);
    Real const face = MOL::hydro_mol_xedge_state(2,0,0,0,line(q,4),line(face_values,3),
                                                 &periodic,-1,4,false,true);
    // -div(v*q_face) + q*div(v), paired with the two cell adjoints.
    return speed * (right*(face-q[2]) - left*(face-q[1]));
}

Real normalPair(Real const* q, Real weight)
{
    // Periodic branch of MOL::ExtrapVelToFacesBox's normal Riemann solver.
    auto const v = line(q,4);
    Real const left  = q[1] + 0.5*amrex_calc_xslope(1,0,0,0,2,v);
    Real const right = q[2] - 0.5*amrex_calc_xslope(2,0,0,0,2,v);
    Real const avg = 0.5*(left+right);
    if (left >= 0 || right <= 0) {
        if (avg >= small_vel) return weight*left;
        if (avg <= -small_vel) return weight*right;
    }
    return 0;
}

 } // namespace

extern "C" void incflo_dss_flux_vjp(Real const* q, Real* dq, Real left, Real right)
{
    __enzyme_autodiff(reinterpret_cast<void*>(fluxPair), enzyme_dup, q, dq,
                      enzyme_const, left, enzyme_const, right);
}

extern "C" void incflo_dss_normal_vjp(Real const* q, Real* dq, Real weight)
{
    __enzyme_autodiff(reinterpret_cast<void*>(normalPair), enzyme_dup, q, dq, enzyme_const, weight);
}
