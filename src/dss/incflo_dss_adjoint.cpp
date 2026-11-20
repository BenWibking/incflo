#include <incflo.H>
#include <hydro_mol.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Loop.H>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace amrex;

extern "C" AMREX_GPU_DEVICE void incflo_dss_flux_vjp(Real const*, Real*, Real, Real);
extern "C" AMREX_GPU_DEVICE void incflo_dss_normal_vjp(Real const*, Real*, Real);

namespace {
MultiFab cloneField(MultiFab const& in, int ghosts=0)
{
    MultiFab out(in.boxArray(), in.DistributionMap(), 3, ghosts);
    MultiFab::Copy(out,in,0,0,3,0);
    return out;
}

AMREX_GPU_HOST_DEVICE GpuArray<IntVect,4> stencil(IntVect const& right, int dir, Box const& domain)
{
    GpuArray<IntVect,4> points{right,right,right,right};
    for (int s=0; s<4; ++s) {
        int& i = points[s][dir];
        i += s-2;
        if (i < domain.smallEnd(dir)) i += domain.length(dir);
        if (i > domain.bigEnd(dir)) i -= domain.length(dir);
    }
    return points;
}

void requireLayout(MultiFab const& v, Geometry const& g)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ParallelDescriptor::NProcs()==1 &&
        v.boxArray().size()==1 && v.boxArray()[0]==g.Domain() &&
        g.isAllPeriodic() && g.Coord()==0 && g.Domain().shortside()>=4,
        "DSS adjoint requires a periodic Cartesian single box on one MPI rank (amr.max_grid_size >= n_cell)");
}

void readBytes(std::string const& path, MultiFab& field)
{
    std::ifstream in(path, std::ios::binary);
    auto& fab = field[0];
    Gpu::PinnedVector<Real> host(fab.box().numPts()*3);
    in.read(reinterpret_cast<char*>(host.data()), host.size()*sizeof(Real));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(in, "Incomplete DSS velocity file: "+path);
    Gpu::htod_memcpy(fab.dataPtr(),host.data(),host.size()*sizeof(Real));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(in && in.peek()==std::ifstream::traits_type::eof() &&
        field.is_finite(), "Invalid DSS velocity file (expected 3*nz*ny*nx native float64 values): "+path);
}

void writeField(std::string const& path, MultiFab const& v)
{
    auto packed = cloneField(v);
    auto const& fab = packed[0];
    std::ofstream out(path, std::ios::binary);
    Gpu::PinnedVector<Real> host(fab.box().numPts()*3);
    Gpu::dtoh_memcpy(host.data(),fab.dataPtr(),host.size()*sizeof(Real));
    out.write(reinterpret_cast<char const*>(host.data()), host.size()*sizeof(Real));
    out.close();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(out, "Could not write DSS field: "+path);
}
} // namespace

bool incflo::dssReadControl(int lev)
{
    std::string path;
    if (!ParmParse("dss").query("control",path)) return false;
    bool enabled=false;
    ParmParse("dss").query("enabled",enabled);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(enabled && lev==0 && max_level==0 && m_probtype==0,
                                     "dss.control requires dss.enabled, probtype=0 and one level");
    auto& velocity = m_leveldata[0]->velocity;
    requireLayout(velocity, geom[0]);
    MultiFab packed(grids[0], dmap[0], 3, 0);
    readBytes(path,packed);
    MultiFab::Copy(velocity,packed,0,0,3,0);
    return true;
}

void incflo::dssStart()
{
    ParmParse pp("dss");
    pp.query("enabled",m_dss_enabled);
    if (!m_dss_enabled) return;
    pp.query("gradient",m_dss_gradient);
    auto const& u = m_leveldata[0]->velocity;
    requireLayout(u,geom[0]);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_level==0 && m_restart_file.empty() && m_cur_time==0 &&
        m_advection_type=="MOL" && m_constant_density && m_ro_0==1 && m_mu==1 &&
        m_fluid_model==FluidModel::Newtonian && m_diff_type==DiffusionType::Explicit &&
        !m_use_cc_proj && !m_advect_momentum && !m_use_divu_source && !m_use_boussinesq &&
        !m_advect_tracer && !m_use_temperature && !m_steady_state &&
        !m_mac_mixed_precision && !m_nodal_mixed_precision && !use_tensor_correction &&
        m_fixed_dt>0 && m_stop_time>0 && m_stop_time<1,
        "DSS adjoint requires fixed-dt, constant unit density/viscosity, unforced periodic MOL with explicit diffusion and nodal projection");
    for (int d=0; d<3; ++d) {
        AMREX_ALWAYS_ASSERT(m_gravity[d]==0 && m_delp[d]==0);
    }
    m_dss_initial = cloneField(u);
}

void incflo::dssRecord()
{
    m_dss_stages.push_back(cloneField(m_leveldata[0]->velocity));
}

// With periodic boundaries, constant density and equal cell volumes, D=-G^T.
// The nodal and MAC maps I-G L^{-1}D are symmetric. Apply the native
// projectors in reverse to solver tolerance, rather than taping multigrid.
void incflo::dssProject(MultiFab& v) const
{
    AMREX_ALWAYS_ASSERT(v.nGrow()>=1);
    v.setBndry(0);
    v.FillBoundary(geom[0].periodicity());
    LPInfo info;
    info.setMaxCoarseningLevel(m_nodal_mg_max_coarsening_level);
    Hydro::NodalProjector p({&v},1.0,Geom(0,0),info);
    p.setDomainBC(get_projection_bc(Orientation::low),get_projection_bc(Orientation::high));
    p.project(m_nodal_mg_rtol, m_nodal_mg_atol);
}

void incflo::dssProjectFaces(Array<MultiFab,3>& faces) const
{
    LPInfo info;
    info.setMaxCoarseningLevel(m_mac_mg_max_coarsening_level);
    Vector<Array<MultiFab*,3>> v{GetArrOfPtrs(faces)};
    Hydro::MacProjector p(v,1.0,Geom(0,0),info);
    p.setDomainBC(get_mac_projection_bc(Orientation::low),get_mac_projection_bc(Orientation::high));
    p.project(m_mac_mg_rtol,m_mac_mg_atol);
}

MultiFab incflo::dssRhsTranspose(MultiFab const& state, MultiFab const& weight)
{
    auto v = cloneField(state,2);
    v.FillBoundary(geom[0].periodicity());
    auto w = cloneField(weight,1);
    w.FillBoundary(geom[0].periodicity());
    MultiFab result(grids[0],dmap[0],3,1);
    MultiFab eta(grids[0],dmap[0],1,1);
    eta.setVal(m_mu);
    // The constant-coefficient periodic viscous stress operator is symmetric.
    compute_divtau({&result},{&w},get_density_new_const(),{&eta});

    Array<MultiFab,3> faces, face_bar;
    for (int d=0; d<3; ++d) {
        auto ba = convert(grids[0], IntVect::TheDimensionVector(d));
        faces[d].define(ba,dmap[0],1,0);
        face_bar[d].define(ba,dmap[0],1,0);
    }
    MOL::ExtrapVelToFaces(v,faces[0],faces[1],faces[2],geom[0],
                         get_velocity_bcrec(),get_velocity_bcrec_device_ptr());
    dssProjectFaces(faces);
    auto const u = v[0].const_array();
    auto const adj = w[0].const_array();
    auto const grad = result[0].array();
    Box const& domain = geom[0].Domain();
    for (int d=0; d<3; ++d) {
        Real const inv_dx=geom[0].InvCellSize(d);
        auto const speed=faces[d][0].const_array();
        auto const bar=face_bar[d][0].array();
        // Each periodic face has one independent value, including the seam.
        ParallelFor(domain,[=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
            IntVect p(i,j,k);
            auto const ix=stencil(p,d,domain);
            Real da=0;
            for (int n=0; n<3; ++n) {
                Real q[5]={u(ix[0],n),u(ix[1],n),u(ix[2],n),u(ix[3],n),speed(p)};
                Real dq[5]={};
                incflo_dss_flux_vjp(q,dq,adj(ix[1],n)*inv_dx,adj(p,n)*inv_dx);
                for (int s=0; s<4; ++s) Gpu::Atomic::AddNoRet(&grad(ix[s],n),dq[s]);
                da+=dq[4];
            }
            bar(p)=da;
            if (p[d]==domain.smallEnd(d)) {
                p[d]=domain.bigEnd(d)+1;
                bar(p)=da;
            }
        });
    }
    dssProjectFaces(face_bar);
    for (int d=0; d<3; ++d) {
        auto const bar=face_bar[d][0].const_array();
        ParallelFor(domain,[=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
            IntVect p(i,j,k);
            auto const ix=stencil(p,d,domain);
            Real q[4]={u(ix[0],d),u(ix[1],d),u(ix[2],d),u(ix[3],d)};
            Real dq[4]={};
            incflo_dss_normal_vjp(q,dq,bar(p));
            for (int s=0; s<4; ++s) Gpu::Atomic::AddNoRet(&grad(ix[s],d),dq[s]);
        });
    }
    return result;
}

void incflo::dssFinish()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(m_cur_time-m_stop_time)<1.e-12 && m_nstep>0 &&
        std::abs(m_nstep*m_fixed_dt-m_stop_time)<1.e-12,
        "Set fixed_dt to the one-cycle physical interval divided by an integer number of steps");
    Real radius=50;
    ParmParse pp("dss");
    pp.query("radius",radius);
    AMREX_ALWAYS_ASSERT(radius>0);
    Real const scale=std::sqrt(1-m_stop_time);
    auto const dx=geom[0].CellSizeArray(), lo=geom[0].ProbLoArray();
    Box const& domain=geom[0].Domain();
    // Only the two endpoint fields and the terminal loss weights cross the
    // host/device boundary. All stages, reconstructions and solves stay on device.
    FArrayBox host_first(domain,3,The_Pinned_Arena()), host_last(domain,3,The_Pinned_Arena());
    FArrayBox host_terminal(domain,3,The_Pinned_Arena()), host_direct(domain,3,The_Pinned_Arena());
    auto const packed_last=cloneField(m_leveldata[0]->velocity);
    auto const elements=domain.numPts()*3;
    auto const bytes=elements*sizeof(Real);
    Gpu::dtoh_memcpy(host_first.dataPtr(),m_dss_initial[0].dataPtr(),bytes);
    Gpu::dtoh_memcpy(host_last.dataPtr(),packed_last[0].dataPtr(),bytes);
    std::fill_n(host_terminal.dataPtr(),elements,0.);
    std::fill_n(host_direct.dataPtr(),elements,0.);
    auto const first=host_first.const_array(), last=host_last.const_array();
    auto const terminal=host_terminal.array(), direct=host_direct.array();
    Real square=0, norm=0;
    LoopOnCpu(domain,[&](int i,int j,int k) {
        IntVect index(i,j,k), base;
        Real r2=0;
        std::array<Real,3> frac{};
        for (int d=0; d<3; ++d) {
            Real const y=lo[d]+(index[d]-domain.smallEnd(d)+0.5)*dx[d];
            r2+=y*y;
            Real const pos=(scale*y-lo[d])/dx[d]-0.5+domain.smallEnd(d);
            base[d]=static_cast<int>(std::floor(pos));
            frac[d]=pos-base[d];
        }
        if (r2>radius*radius) return;
        AMREX_ALWAYS_ASSERT(domain.contains(base) && domain.contains(base+IntVect(1)));
        for (int comp=0; comp<3; ++comp) {
            Real sample=0;
            for (int z=0; z<2; ++z) for (int y=0; y<2; ++y) for (int x=0; x<2; ++x) {
                Real const w=(x?frac[0]:1-frac[0])*(y?frac[1]:1-frac[1])*(z?frac[2]:1-frac[2]);
                sample+=w*last(base+IntVect(x,y,z),comp);
            }
            Real const u0=first(index,comp), e=scale*sample-u0;
            square+=e*e;
            norm+=u0*u0;
            if (m_dss_gradient) {
                direct(index,comp)=-e;
                for (int z=0; z<2; ++z) for (int y=0; y<2; ++y) for (int x=0; x<2; ++x) {
                    Real const w=(x?frac[0]:1-frac[0])*(y?frac[1]:1-frac[1])*(z?frac[2]:1-frac[2]);
                    terminal(base+IntVect(x,y,z),comp)+=scale*w*e;
                }
            }
        }
    });
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(norm>0 && std::isfinite(norm) && std::isfinite(square),
                                     "DSS relative objective requires finite nonzero core velocity");
    Real const ratio=square/norm;
    if (m_dss_gradient) {
        // Reverse the normalized loss, including the dependence of the denominator on U0.
        LoopOnCpu(domain,[&](int i,int j,int k) {
            IntVect p(i,j,k);
            Real r2=0;
            for (int d=0;d<3;++d) {
                Real const y=lo[d]+(p[d]-domain.smallEnd(d)+0.5)*dx[d];
                r2+=y*y;
            }
            if (r2<=radius*radius) {
                for (int n=0;n<3;++n) direct(p,n)=(direct(p,n)-ratio*first(p,n))/norm;
            }
        });
        MultiFab initial_bar(grids[0],dmap[0],3,0);
        Gpu::htod_memcpy(initial_bar[0].dataPtr(),host_terminal.dataPtr(),bytes);
        auto lambda=cloneField(initial_bar,1);
        Gpu::htod_memcpy(initial_bar[0].dataPtr(),host_direct.dataPtr(),bytes);
        lambda.mult(1/norm,0,3);
        AMREX_ALWAYS_ASSERT(m_dss_stages.size()==2*static_cast<std::size_t>(m_nstep));
        // up=P(u+hF(u)), unew=P(u+h/2*(F(u)+F(up))). Old pressure cancels
        // between the force and the nonincremental projection at constant density.
        for (int step=m_nstep-1;step>=0;--step) {
            dssProject(lambda);
            auto pred_bar=dssRhsTranspose(m_dss_stages.back(),lambda);
            m_dss_stages.pop_back();
            pred_bar.mult(0.5*m_fixed_dt,0,3);
            dssProject(pred_bar);
            auto f_bar=cloneField(pred_bar);
            MultiFab::Saxpy(f_bar,0.5,lambda,0,0,3,0);
            auto old_bar=dssRhsTranspose(m_dss_stages.back(),f_bar);
            m_dss_stages.pop_back();
            MultiFab::Add(lambda,pred_bar,0,0,3,0);
            MultiFab::Saxpy(lambda,m_fixed_dt,old_bar,0,0,3,0);
        }
        MultiFab::Add(lambda,initial_bar,0,0,3,0);
        if (m_do_initial_proj) dssProject(lambda);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lambda.is_finite(0,3,0),"Nonfinite DSS gradient");
        writeField("dss_gradient.bin",lambda);
    }
    std::ofstream out("dss_result.txt");
    out << std::setprecision(17) << 0.5*ratio << ' ' << std::sqrt(ratio) << ' '
        << m_cur_time << ' ' << m_nstep << '\n';
    out.close();
    AMREX_ALWAYS_ASSERT(out);
    Print() << "DSS relative velocity return error = " << std::sqrt(ratio) << "\n";
}
