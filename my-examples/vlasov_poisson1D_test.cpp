#include <lr/lr.hpp>
#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <lr/coefficients.hpp>
#include <generic/kernels.hpp>
#include <generic/timer.hpp>
#include <generic/fft.hpp>
#include <generic/netcdf.hpp>
#include <generic/kernels.hpp>

#include <generic/optparser.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <string>
#include <stdexcept>

bool CPU;

template<size_t d> using mind = array<Index,d>;
template<size_t d> using mfp  = array<double,d>; 
using mat  = Ensign::multi_array<double,2>;
using vec  = Ensign::multi_array<double,1>;
using cmat = Ensign::multi_array<complex<double>,2>;
using cvec = Ensign::multi_array<complex<double>,1>;

template<class func>
void componentwise_vec_omp(const mind<1>& N, func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index i = 0; i < N[0]; i++){
    Index idx = i;
    F(idx, {i});
  }
};

template<class func>
void componentwise_mat_omp(const mind<1>& N, 
                          const Index& r,
                          func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index i = 0; i < N[0]; i++) {
    for (Index ri = 0; ri < r; ri++)
    {
      Index idx = i;
      F(idx, {i}, ri);
    };
  };
};

template<size_t d>
struct grid_info {
  Index r;
  mind<d> N_xx; // x维度网格点数量
  mind<d> N_vv; // v维度网格点数量
  mfp<2*d> lim_xx, lim_vv; // x维度和v维度的边界
  mfp<d>   h_xx, h_vv; // x维度和v维度的网格间距
  Index dxx_mult, dvv_mult; // x维度和v维度的网格点总数
  // Index dxxh_mult, dvvh_mult; 

  grid_info(Index r_, mind<d> N_xx_, mind<d> N_vv_, 
            mfp<2*d> lim_xx_, mfp<2*d> lim_vv_)
    : r(r_), N_xx(N_xx_), N_vv(N_vv_), 
              lim_xx(lim_xx_), lim_vv(lim_vv_) 
  {

    // compute h_xx and h_vv
    dxx_mult = 1;
    dvv_mult = 1;
    for(size_t ii = 0; ii < d; ii++){
      Index jj = 2*ii;
      h_xx[ii] = (lim_xx[jj+1]-lim_xx[jj])/ N_xx[ii];
      h_vv[ii] = (lim_vv[jj+1]-lim_vv[jj])/ N_vv[ii];

      dxx_mult *= N_xx[ii];
      dvv_mult *= N_vv[ii];
    }
  
    // dxxh_mult = (N_xx[0]/2 + 1) * (dxx_mult/N_xx[0]);
    // dvvh_mult = (N_vv[0]/2 + 1) * (dvv_mult/N_vv[0]);
  }

  Index lin_idx_x(mind<d> i) const {
    Index idx=0, stride=1;
    for(size_t k=0;k<d;k++) {
      idx += stride*i[k];
      stride *= N_xx[k];
    }
    return idx;
  }
  
  Index lin_idx_v(mind<d> i) const {
    Index idx=0, stride=1;
    for(size_t k=0;k<d;k++) {
      idx += stride*i[k];
      stride *= N_vv[k];
    }
    return idx;
  }

  double x(size_t k, Index i) const {
    return lim_xx[2*k] + i*h_xx[k];
  }

  mfp<d> x(mind<d> i) const {
    mfp<d> out;
    for(size_t k=0;k<d;k++)
      out[k] = x(k, i[k]);
    return out;
  }

  double v(size_t k, Index i) const {
    return lim_vv[2*k] + i*h_vv[k];
  }

  mfp<d> v(mind<d> i) const {
    mfp<d> out;
    for(size_t k=0;k<d;k++)
      out[k] = v(k, i[k]);
    return out;
  }
};

struct coeff_C {
coeff_C(stloc sl, grid_info<1> _mesh) : 
        mesh(_mesh)
{
  weights_plus.resize({mesh.dvv_mult});
  weights_neg.resize({mesh.dvv_mult});
  dV_neg_wind.resize({mesh.dvv_mult, mesh.r});
  dV_plus_wind.resize({mesh.dvv_mult, mesh.r});

  // Initialize we
  vec v({mesh.dvv_mult}, sl);
  componentwise_vec_omp(mesh.N_vv, [this, &v](Index idx, mind<1> i) {
    v(idx) = mesh.v(0, i[0]);
  });

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < mesh.dvv_mult; j++) {
    weights_plus(j) = (v(j) > 0) ? v(j) * mesh.h_vv[0] : 0;
    weights_neg(j) = (v(j) <= 0) ? v(j) * mesh.h_vv[0] : 0;
  };
};

void operator()(mat& V, mat& C1_plus_wind, mat& C1_neg_wind,
               mat& C2_plus_wind, mat& C2_neg_wind, 
               const Ensign::Matrix::blas_ops& blas) {
  // C1
  coeff(V, V, weights_plus, C1_plus_wind, blas);
  coeff(V, V, weights_neg, C1_neg_wind, blas);

  // compute dV
  componentwise_mat_omp(
    mesh.N_vv,
    mesh.r,
    [this, &V](Index idx, mind<1> i, Index ri) {
      const Index n = mesh.dvv_mult;
      const double hv = mesh.h_vv[0];

      // 网格外的虚拟点取 0
      const double center = V(idx, ri);
      const double left  = (idx > 0)     ? V(idx - 1, ri) : 0.0;
      const double right = (idx + 1 < n) ? V(idx + 1, ri) : 0.0;

      dV_plus_wind(idx, ri) = (center - left) / hv;
      dV_neg_wind(idx, ri)  = (right - center) / hv;
  }); 

  // C2
  coeff(V, dV_plus_wind, mesh.h_vv[0], C2_plus_wind, blas);
  coeff(V, dV_neg_wind, mesh.h_vv[0], C2_neg_wind, blas);
};

private:

  grid_info<1> mesh;
  vec weights_plus;
  vec weights_neg;
  mat dV_plus_wind;
  mat dV_neg_wind;

};

struct coeff_D {
coeff_D(stloc sl, grid_info<1> _mesh) : 
        mesh(_mesh)
{
  weights_plus.resize({mesh.dxx_mult});
  weights_neg.resize({mesh.dxx_mult});
  dX_neg_wind.resize({mesh.dxx_mult, mesh.r});
  dX_plus_wind.resize({mesh.dxx_mult, mesh.r});
};

void operator()(mat& X, const vec& E, 
                mat& D1_plus_wind, mat& D1_neg_wind, 
                mat& D2_plus_wind, mat& D2_neg_wind, 
                const Ensign::Matrix::blas_ops& blas) {
  // Initialize we
  vec x({mesh.dxx_mult});
  componentwise_vec_omp(mesh.N_xx, [this, &x](Index idx, mind<1> i) {
    x(idx) = mesh.x(0, i[0]);
  });

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < mesh.dxx_mult; j++) {
    weights_plus(j) = (E(j) > 0) ? E(j) * mesh.h_xx[0] : 0;
    weights_neg(j) = (E(j) <= 0)? E(j) * mesh.h_xx[0] : 0;
  };

  // D1
  coeff(X, X, weights_plus, D1_plus_wind, blas);
  coeff(X, X, weights_neg, D1_neg_wind, blas);

  // compute dX
  componentwise_mat_omp(
    mesh.N_xx,
    mesh.r,
    [this, &X](Index idx, mind<1> i, Index ri) {
      const Index n = mesh.dxx_mult;
      const double hx = mesh.h_xx[0];

      // X 取周期边界条件
      const double center = X(idx, ri);
      const double left  = (idx > 0)     ? X(idx - 1, ri) : X(n-1, ri);
      const double right = (idx + 1 < n) ? X(idx + 1, ri) : X(0, ri);

      dX_plus_wind(idx, ri) = (center - left) / hx;
      dX_neg_wind(idx, ri)  = (right - center) / hx;
  }); 

  // C2
  coeff(X, dX_plus_wind, mesh.h_xx[0], D2_plus_wind, blas);
  coeff(X, dX_neg_wind, mesh.h_xx[0], D2_neg_wind, blas);
};

private:

  grid_info<1> mesh;
  vec weights_plus;
  vec weights_neg;
  mat dX_plus_wind;
  mat dX_neg_wind;

};

struct electric_field {
  electric_field(stloc sl, grid_info<1> _mesh)
    : mesh(_mesh), solver(mesh.dxx_mult), int_V({mesh.r}), phi({mesh.dxx_mult}) {
    const Index n = mesh.dxx_mult;
    const double hx = mesh.h_xx[0];
    const double inv_hx2 = 1.0 / (hx * hx);
    for (Index i = 0; i < n; ++i) {
      for (Index j = 0; j < n; ++j)
        solver.A(i, j) = 0.0;
      // -phi_xx with periodic neighbors. Use += because they coincide for n <= 2.
      solver.A(i, i) += 2.0 * inv_hx2;
      solver.A(i, (i + n - 1) % n) -= inv_hx2;
      solver.A(i, (i + 1) % n) -= inv_hx2;
    }
    // Add the discrete zero-integral constraint to the middle equation.
    for (Index j = 0; j < n; ++j)
      solver.A(n / 2, j) += 1.0 / hx;
    solver.lu();
  }

  void operator()(const mat& K, const mat& V, vec& E, const Ensign::Matrix::blas_ops& blas) {
    // -phi_xx = 1 - integral f dv, E = -phi_x.
    integrate(V, -mesh.h_vv[0], int_V, blas);
    blas.matvec(K, int_V, phi);
    phi += 1.0;

    // The periodic Poisson equation requires a zero-mean right-hand side.
    double mean = 0.0;
    const Index n = mesh.dxx_mult;
    for (Index i = 0; i < n; ++i)
      mean += phi(i);
    mean /= static_cast<double>(n);
    for (Index i = 0; i < n; ++i)
      phi(i) -= mean;

    solver.solve(phi);
    const double inv_2hx = 0.5 / mesh.h_xx[0];
    for (Index i = 0; i < n; ++i)
      E(i) = -(phi((i + 1) % n) - phi((i + n - 1) % n)) * inv_2hx;
  }

private:
  grid_info<1> mesh;
  Ensign::Matrix::lu_solver<double> solver;
  vec int_V, phi;
};

double setdt(const grid_info<1>& mesh, 
            const double& cfl,
            const vec& E)
{
  const double& hx = mesh.h_xx[0];
  const double& hv = mesh.h_vv[0];
  // Bound the transport speed by the velocity-domain endpoints.
  const double vmax = std::max(std::abs(mesh.lim_vv[0]), std::abs(mesh.lim_vv[1]));
  double emax = 0.0;
  for (Index i = 0; i < mesh.dxx_mult; ++i) {
    emax = std::max(emax, std::abs(E(i)));
  }

  // Combined CFL condition: dt * (|v|max / hx + |E|max / hv) <= cfl.
  const double rate = vmax / hx + emax / hv;
  return rate > 0.0 ? cfl / rate : std::numeric_limits<double>::infinity();
};

struct KSL_K_step {
  KSL_K_step(stloc _sl, grid_info<1> _mesh, const Ensign::Matrix::blas_ops* _blas)
    : mesh(_mesh), blas(_blas),
      dK_plus_wind({mesh.dxx_mult, mesh.r}),
      dK_neg_wind({mesh.dxx_mult, mesh.r}),
      transport_neg({mesh.dxx_mult, mesh.r}), rhs({mesh.dxx_mult, mesh.r}) {
  }

  void operator()(double dt, mat& K, const vec& E, 
                  const mat& C1_plus_wind, const mat& C1_neg_wind,
                  const mat& C2_plus_wind, const mat& C2_neg_wind) {
    // Split the velocity integral before projection: v > 0 uses the
    // backward difference, and v <= 0 uses the forward difference.
    const double inv_hx = 1.0 / mesh.h_xx[0];
    componentwise_mat_omp(mesh.N_xx, mesh.r,
      [this, &K, inv_hx](Index i, mind<1>, Index j) {
        const Index left = (i + mesh.dxx_mult - 1) % mesh.dxx_mult;
        const Index right = (i + 1) % mesh.dxx_mult;
        dK_plus_wind(i, j) = (K(i, j) - K(left, j)) * inv_hx;
        dK_neg_wind(i, j) = (K(right, j) - K(i, j)) * inv_hx;
      });
    // K * C^T
    blas->matmul_transb(dK_plus_wind, C1_plus_wind, rhs);
    blas->matmul_transb(dK_neg_wind, C1_neg_wind, transport_neg);
    // componentwise_mat_omp(mesh.N_xx, mesh.r,
    //   [this](Index i, mind<1>, Index j) {
    //     rhs(i, j) = - rhs(i, j) - transport_neg(i, j);
    //   });
    rhs = rhs * (-1.e0) - transport_neg;

    // f_t + v f_x - E f_v = 0, so velocity advection has speed -E.
    // Positive E uses the forward derivative; negative E uses the backward one.
    componentwise_mat_omp(mesh.N_xx, mesh.r,
      [this, &K, &E, &C2_plus_wind, &C2_neg_wind](Index i, mind<1>, Index j) {
        const mat& C2 = E(i) >= 0.0 ? C2_neg_wind : C2_plus_wind;
        double source = 0.0;
        for (Index k = 0; k < mesh.r; ++k)
          source += K(i, k) * C2(j, k);
        rhs(i, j) += E(i) * source;
      });

    // Keep K unchanged until the complete right-hand side has been assembled.
    // componentwise_mat_omp(mesh.N_xx, mesh.r,
    //   [this, &K, dt](Index i, mind<1>, Index j) {
    //     K(i, j) += dt * rhs(i, j);
    //   });
    K += rhs * dt;
  }

private:
  grid_info<1> mesh;
  const Ensign::Matrix::blas_ops* blas;
  mat dK_plus_wind, dK_neg_wind, transport_neg, rhs;
};

struct KSL_S_step {
  KSL_S_step(stloc _sl, grid_info<1> _mesh, const Ensign::Matrix::blas_ops* _blas)
    : mesh(_mesh), blas(_blas),
      transport_v_plus({mesh.r, mesh.r}), 
      transport_v_neg({mesh.r, mesh.r}), 
      transport_E_plus({mesh.r, mesh.r}), 
      transport_E_neg({mesh.r, mesh.r}), 
      rhs({mesh.r, mesh.r}) {
  }

  void operator()(double dt, mat& S, const vec& E, 
                  const mat& C1_plus_wind, const mat& C1_neg_wind,
                  const mat& C2_plus_wind, const mat& C2_neg_wind,
                  const mat& D1_plus_wind, const mat& D1_neg_wind,
                  const mat& D2_plus_wind, const mat& D2_neg_wind) {    
    // The S substep is minus the projection of the Vlasov right-hand side.
    // D2+/- use backward/forward x differences and C1+/- contain v > 0 / v <= 0.
    auto product = [this, &S](const mat& D, const mat& C, mat& out) {
      // D * S * C^T
      blas->matmul_transb(S, C, rhs);
      blas->matmul(D, rhs, out);
    };
    product(D2_plus_wind, C1_plus_wind, transport_v_plus);
    product(D2_neg_wind, C1_neg_wind, transport_v_neg);
    // Velocity advection speed is -E: E > 0 uses forward v differences.
    product(D1_plus_wind, C2_neg_wind, transport_E_plus);
    product(D1_neg_wind, C2_plus_wind, transport_E_neg);

    S += (transport_v_plus + transport_v_neg - transport_E_plus - transport_E_neg) * dt;
  }

private:
  grid_info<1> mesh;
  const Ensign::Matrix::blas_ops* blas;
  mat transport_v_plus, transport_v_neg;
  mat transport_E_plus, transport_E_neg;
  mat rhs;
};

struct KSL_L_step {
  KSL_L_step(stloc _sl, grid_info<1> _mesh, const Ensign::Matrix::blas_ops* _blas)
    : mesh(_mesh), blas(_blas),
      dL_plus_wind({mesh.dvv_mult, mesh.r}),
      dL_neg_wind({mesh.dvv_mult, mesh.r}),
      transport_neg({mesh.dvv_mult, mesh.r}), 
      rhs({mesh.dvv_mult, mesh.r}) {
  }

  void operator()(double dt, mat& L, 
                  const mat& D1_plus_wind, const mat& D1_neg_wind,
                  const mat& D2_plus_wind, const mat& D2_neg_wind) {
    // Split the velocity integral before projection: v > 0 uses the
    // backward difference, and v <= 0 uses the forward difference.
    const double inv_hv = 1.0 / mesh.h_vv[0];
    const Index n = mesh.dvv_mult;
    componentwise_mat_omp(mesh.N_vv, mesh.r,
      [this, &L, &n ,&inv_hv](Index i, mind<1>, Index j) {
        const double center = L(i, j);
        const double left  = (i > 0)     ? L(i - 1, j) : 0.0;
        const double right = (i + 1 < n) ? L(i + 1, j) : 0.0;

        dL_plus_wind(i, j) = (center - left) * inv_hv;
        dL_neg_wind(i, j) = (right - center) * inv_hv;
      });
    // L * D1^T
    blas->matmul_transb(dL_plus_wind, D1_neg_wind, rhs);
    blas->matmul_transb(dL_neg_wind, D1_plus_wind, transport_neg);
    rhs += transport_neg;

    // f_t + v f_x - E f_v = 0, so velocity advection has speed -E.
    // Positive E uses the forward derivative; negative E uses the backward one.
    componentwise_mat_omp(mesh.N_vv, mesh.r,
      [this, &L, &D2_plus_wind, &D2_neg_wind](Index i, mind<1>, Index j) {
        double v = mesh.v(0, i);
        const mat& D2 = mesh.v(0, i) >= 0.0 ? D2_plus_wind : D2_neg_wind;
        double source = 0.0;
        for (Index k = 0; k < mesh.r; ++k)
          source += L(i, k) * D2(j, k);
        rhs(i, j) -= v * source;
      });

    // Keep L unchanged until the complete right-hand side has been assembled.
    L += rhs * dt;
  }

private:
  grid_info<1> mesh;
  const Ensign::Matrix::blas_ops* blas;
  mat dL_plus_wind, dL_neg_wind, transport_neg, rhs;
};

int main(int argc, char** argv)
{
  std::string OutputDir = "result/vlasov_poisson1D_test";
  std::filesystem::create_directories(OutputDir);
  constexpr int dim = 1;
  constexpr int dim2 = 2 * dim;
  std::string device = "CPU";
  std::string problem = "ll";
  double cfl = 0.8e0;
  double x1 = 0.e0;
  double x2 = 4.e0 * M_PI;
  double v1 = -6.e0;
  double v2 = 6.e0;
  int rank = 10;
  int nx = 100;
  int nv = 100;
  int omp_threads = 5;
  int snapshots = 0;
  int plot = 0;
  int output_gap = 10;

  vec Tplot;

  Ensign::OptionsParser options(argc, argv);
  options.AddOption(&device, "-device", "--device", "Simulation device only by <CPU> now");
  options.AddOption(&problem, "-problem", "--problem", "Initial condition (ll or ts)");
  options.AddOption(&cfl, "-cfl", "--cfl", "CFL condition numbers");
  options.AddOption(&rank, "-r", "--rank", "Simulation rank");
  options.AddOption(&nx, "-nx", "--nx", "Grid point counts in space as a quoted list");
  options.AddOption(&nv, "-nv", "--nv", "Grid point counts in velocity as a quoted list");
  options.AddOption(&omp_threads, "-omp_threads", "--omp_threads",
                    "OpenMP threads (-1 uses half the available processors)");
  options.AddOption(&snapshots, "-snapshots", "--snapshots", "Number of snapshots to write");
  options.AddOption(&plot, "-plot", "--plot", "whether to plot (>0 means plot while Tplot)");
  options.AddOption(&Tplot, "-Tplot", "--Tplot", "The plot time while simulating");
  options.AddOption(&output_gap, "-output", "--outputgap", "Output interval in time steps.");
  options.ParseCheck();


  #ifdef __OPENMP__
  int num_threads = omp_threads;
  if(num_threads == -1)
    num_threads = omp_get_num_procs()/2;
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    if(omp_get_thread_num()==0){
      cout << "Number of threads: " << omp_get_num_threads() << endl;
    }
  }
  #endif

  mind<dim> N_xx = {nx};
  mind<dim> N_vv = {nv};  
  mfp<dim2> lim_xx = {x1, x2}; // Limits for box [x1,x2] {x1,x2}
  mfp<dim2> lim_vv = {v1, v2}; // Limits for box [v1,v2] {v1,v2}
  Index r = rank;
  grid_info<dim> mesh(r, N_xx, N_vv, lim_xx, lim_vv);

  vec fx({mesh.dxx_mult});
  componentwise_vec_omp(mesh.N_xx, [&fx, &mesh](Index idx, mind<dim> i) {
    double delta1 = 0.01;
    double kappa1 = 0.5;
    mfp<dim> x  = mesh.x(i);
    fx(idx) = 1.0 + delta1 * std::cos(kappa1*x[0]);
  });

  vec fv({mesh.dvv_mult});
  componentwise_vec_omp(mesh.N_vv, [&fv, &mesh](Index idx, mind<dim> i) {
    mfp<dim> v  = mesh.v(i);
    fv(idx) = (1.0/(sqrt(pow(2*M_PI, dim)))) * exp(-(std::pow(v[0], 2)) / 2.0);
  });
  
  std::vector<const double*> X0, V0;
  X0.push_back(fx.begin());
  V0.push_back(fv.begin());

  #ifndef __CUDA__
  CPU = true;
  #else
  if(device == "CPU")
    CPU = true;
  else if(device == "GPU")
    CPU = false;
  else {
    cout << "ERROR: device " << device << " not found." << endl;
    exit(1);
  }
  #endif

  Ensign::Matrix::blas_ops blas(!CPU);
  Ensign::orthogonalize QR(&blas);

  stloc sl = (device == "CPU") ? stloc::host : stloc::device;
  Ensign::lr2<double> low_rank_solver(mesh.r, {mesh.dxx_mult, mesh.dvv_mult}, sl);
  std::function<double(double*,double*)> ip_xx = 
    Ensign::inner_product_from_const_weight(mesh.h_xx[0], mesh.dxx_mult);
  std::function<double(double*,double*)> ip_vv = 
    Ensign::inner_product_from_const_weight(mesh.h_vv[0], mesh.dvv_mult);
  initialize(low_rank_solver, X0, V0, ip_xx, ip_vv, blas);

  // C1, C2 compute preparation
  coeff_C compute_C(sl, mesh);
  mat C1_plus_wind, C1_neg_wind;
  mat C2_plus_wind, C2_neg_wind;
  C1_plus_wind.resize({mesh.r, mesh.r});
  C1_neg_wind.resize({mesh.r, mesh.r});
  C2_plus_wind.resize({mesh.r, mesh.r});
  C2_neg_wind.resize({mesh.r, mesh.r});

  // E compute preparation
  vec E({mesh.dxx_mult});
  electric_field compute_E(sl, mesh);
  
  // D1(E), D2 preparation
  coeff_D compute_D(sl, mesh);
  mat D1_plus_wind, D1_neg_wind;
  mat D2_plus_wind, D2_neg_wind;
  D1_plus_wind.resize({mesh.r, mesh.r});
  D1_neg_wind.resize({mesh.r, mesh.r});
  D2_plus_wind.resize({mesh.r, mesh.r});
  D2_neg_wind.resize({mesh.r, mesh.r});

  // KSL step preparation
  KSL_K_step K_step(sl, mesh, &blas);
  KSL_S_step S_step(sl, mesh, &blas);
  KSL_L_step L_step(sl, mesh, &blas);

  double trun = 0.e0;
  double dt = 0.e0;
  int step_count = 0;
  mat X_temp, V_temp, K_temp;
  X_temp.resize({mesh.dxx_mult, mesh.r});
  V_temp.resize({mesh.dvv_mult, mesh.r});
  K_temp.resize({mesh.dxx_mult, mesh.r});
  for (Index idx = 0; idx < Tplot.shape()[0]; ++idx)
  {
    const double tstop = Tplot(idx);
    std::ofstream outFile_f;
    if (plot > 0)
    {
      std::string filename_f = "X" + std::to_string(mesh.dxx_mult) +
                                "_V" + std::to_string(mesh.dvv_mult) +
                                "_r" + std::to_string(mesh.r) +
                                "_o" + std::to_string(1) +
                                "_cfl" + std::to_string(cfl) +
                                "_T" + std::to_string(tstop) +
                                "_f.dat";
      outFile_f.open(OutputDir + "/" + filename_f);
      outFile_f << std::scientific  << std::setprecision(16);
    }
    while (trun < tstop)
    {
      // compute K and store it in low_rank_solver.X
      X_temp = low_rank_solver.X;
      blas.matmul(X_temp, low_rank_solver.S, low_rank_solver.X);

      // based on K and V to compute E
      compute_E(low_rank_solver.X, low_rank_solver.V, E, blas);

      // setdt based on CFL condition number
      dt = std::min(setdt(mesh, cfl, E), tstop - trun);

      // K_step
      compute_C(low_rank_solver.V, C1_plus_wind, C1_neg_wind,
                C2_plus_wind, C2_neg_wind, blas);
      K_step(dt, low_rank_solver.X, E, 
            C1_plus_wind, C1_neg_wind,
            C2_plus_wind, C2_neg_wind);
      compute_E(low_rank_solver.X, low_rank_solver.V, E, blas); // update E based on new K
      QR(low_rank_solver.X, low_rank_solver.S, ip_xx);

      // S_step
      compute_D(low_rank_solver.X, E, D1_plus_wind, D1_neg_wind, 
                D2_plus_wind, D2_neg_wind, blas);
      S_step(dt, low_rank_solver.S, E,
            C1_plus_wind, C1_neg_wind, C2_plus_wind, C2_neg_wind,
            D1_plus_wind, D1_neg_wind, D2_plus_wind, D2_neg_wind);
      blas.matmul(low_rank_solver.X, low_rank_solver.S, K_temp);
      compute_E(K_temp, low_rank_solver.V, E, blas);

      // L_step
      compute_D(low_rank_solver.X, E, D1_plus_wind, D1_neg_wind, 
                D2_plus_wind, D2_neg_wind, blas);
      V_temp = low_rank_solver.V;
      blas.matmul_transb(V_temp, low_rank_solver.S, low_rank_solver.V); // compute L
      L_step(dt, low_rank_solver.V, 
            D1_plus_wind, D1_neg_wind,
            D2_plus_wind, D2_neg_wind);
      QR(low_rank_solver.V, low_rank_solver.S, ip_vv);
      Ensign::Matrix::transpose_inplace(low_rank_solver.S);

      trun += dt;
      step_count++;
      if (output_gap > 0 && step_count % output_gap == 0)
      {
        std::cout << "Trun = " << trun << ", dt = " << dt << std::endl;
      }
    }

    if (plot > 0)
    {
      // full() returns f(x_i, v_j) = (X S V^T)(i, j).
      // Match Ensign::dump: one value per line, x index varying fastest.
      const mat f_full = low_rank_solver.full(blas);
      for (Index j = 0; j < mesh.dvv_mult; ++j)
        for (Index i = 0; i < mesh.dxx_mult; ++i)
          outFile_f << mesh.x(0, i) << " " 
                    << mesh.v(0, j) << " "
                    << f_full(i, j) << "\n";
      outFile_f.close();
    }
  }
  return 0;
};

