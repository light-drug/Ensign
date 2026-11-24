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
#include <string>
#include <stdexcept>

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
void componentwise_vec_fourier_omp(const mind<1>& N, func F) {
  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index i = 0; i < (N[0]/2+1); i++){
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
  weights.resize({mesh.dvv_mult});
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
    weights(j) = v(j) * mesh.h_vv[0];
  };
};

void operator()(mat& V, mat& C1, mat& C2_plus_wind, mat& C2_neg_wind, const Ensign::Matrix::blas_ops& blas) {
  // C1
  coeff(V, V, weights, C1, blas);
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
  vec weights;
  mat dV_plus_wind;
  mat dV_neg_wind;

};

struct coeff_D {
coeff_D(stloc sl, grid_info<1> _mesh) : 
        mesh(_mesh)
{
  weights.resize({mesh.dxx_mult});
  dX_neg_wind.resize({mesh.dxx_mult, mesh.r});
  dX_plus_wind.resize({mesh.dxx_mult, mesh.r});
};

void operator()(mat& X, const vec& E, mat& D1, mat& D2_plus_wind, mat& D2_neg_wind, const Ensign::Matrix::blas_ops& blas) {
  // Initialize we
  vec x({mesh.dxx_mult});
  componentwise_vec_omp(mesh.N_xx, [this, &x](Index idx, mind<1> i) {
    x(idx) = mesh.x(0, i[0]);
  });

  #ifdef __OPENMP__
  #pragma omp parallel for
  #endif
  for(Index j = 0; j < mesh.dxx_mult; j++) {
    weights(j) = E(j) * mesh.h_xx[0];
  };

  // C1
  coeff(X, X, weights, D1, blas);
  // compute dV
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
  vec weights;
  mat dX_plus_wind;
  mat dX_neg_wind;

};

struct electric_field {
  electric_field(stloc sl, grid_info<1> _mesh)
    : mesh(_mesh), solver(mesh.dxx_mult), int_V({mesh.r}), phi({mesh.dxx_mult}) {
    if (sl != stloc::host)
      throw std::invalid_argument("electric_field requires host storage");

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
    if (K.sl != stloc::host || V.sl != stloc::host || E.sl != stloc::host)
      throw std::invalid_argument("electric_field requires host storage");

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

int main(int argc, char** argv)
{
  constexpr int dim = 1;
  constexpr int dim2 = 2 * dim;
  std::string device = "CPU";
  std::string problem = "ts";
  double x1 = 0.e0;
  double x2 = 4.e0 * M_PI;
  double v1 = -6.e0;
  double v2 = 6.e0;
  double final_time = 60.0;
  double deltat = 0.01;
  int rank = 10;
  int nx = 100;
  int nv = 100;
  int omp_threads = 5;
  int snapshots = 0;

  Ensign::OptionsParser options(argc, argv);
  options.AddOption(&device, "-device", "--device", "Simulation device (CPU or GPU)");
  options.AddOption(&problem, "-problem", "--problem", "Initial condition (ll or ts)");
  options.AddOption(&final_time, "-final_time", "--final_time", "Final simulation time");
  options.AddOption(&deltat, "-deltat", "--deltat", "Time step");
  options.AddOption(&rank, "-r", "--rank", "Simulation rank");
  options.AddOption(&nx, "-nx", "--nx", "Grid point counts in space as a quoted list");
  options.AddOption(&nv, "-nv", "--nv", "Grid point counts in velocity as a quoted list");
  options.AddOption(&omp_threads, "-omp_threads", "--omp_threads",
                    "OpenMP threads (-1 uses half the available processors)");
  options.AddOption(&snapshots, "-snapshots", "--snapshots", "Number of snapshots to write");
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

  if (nx <= 0 || nv <= 0 || rank <= 0 || rank > nx || rank > nv) {
    std::cerr << "Require nx > 0, nv > 0 and 1 <= rank <= min(nx, nv)." << std::endl;
    return 1;
  }

  mind<dim> N_xx = {nx};
  mind<dim> N_vv = {nv};  
  mfp<dim2> lim_xx = {x1, x2}; // Limits for box [x1,x2] {x1,x2}
  mfp<dim2> lim_vv = {v1, v2}; // Limits for box [v1,v2] {v1,v2}
  Index r = rank;
  grid_info<dim> mesh(r, N_xx, N_vv, lim_xx, lim_vv);

  vec fx({mesh.dxx_mult});
  double Sx = 0.e0;
  std::cout << "dxx_mult: " << mesh.dxx_mult << std::endl;
  // 此处传入的i为该点位置
  componentwise_vec_omp(mesh.N_xx, [&fx, &mesh](Index idx, mind<dim> i) {
    double delta1 = 0.01;
    double kappa1 = 0.5;
    mfp<dim> x  = mesh.x(i);
    fx(idx) = 1.0 + delta1 * std::cos(kappa1*x[0]);
  });
  for(Index i = 0; i < mesh.N_xx[0]; i++) {
    Sx += mesh.h_xx[0] * fx(i) * fx(i);
  }   
  Sx = std::sqrt(Sx);
  fx /= Sx;

  vec fv({mesh.dvv_mult});
  double Sv = 0.e0;
  std::cout << "dvv_mult: " << mesh.dvv_mult << std::endl;
  componentwise_vec_omp(mesh.N_vv, [&fv, &mesh](Index idx, mind<dim> i) {
    mfp<dim> v  = mesh.v(i);
    fv(idx) = (1.0/(sqrt(pow(2*M_PI, dim)))) * exp(-(std::pow(v[0], 2)) / 2.0);
  });
  for(Index i = 0; i < mesh.N_vv[0]; i++) {
    Sv += mesh.h_vv[0] * fv(i) * fv(i);
  }
  Sv = std::sqrt(Sv);
  fv /= Sv;

  // Ensign stores basis vectors in columns: X is nx x r and V is nv x r.
  mat X({mesh.dxx_mult, mesh.r}), V({mesh.dvv_mult, mesh.r});
  mat S({mesh.r, mesh.r}), Rx({mesh.r, mesh.r}), Rv({mesh.r, mesh.r});
  for (Index i = 0; i < mesh.dxx_mult; ++i)
    X(i, 0) = fx(i);
  for (Index i = 0; i < mesh.dvv_mult; ++i)
    V(i, 0) = fv(i);
  // Omit the unit vector at the largest component to keep the basis independent.
  Index pivot_x = 0, pivot_v = 0;
  for (Index i = 1; i < mesh.dxx_mult; ++i)
    if (std::abs(fx(i)) > std::abs(fx(pivot_x))) pivot_x = i;
  for (Index i = 1; i < mesh.dvv_mult; ++i)
    if (std::abs(fv(i)) > std::abs(fv(pivot_v))) pivot_v = i;
  for (Index j = 1; j < mesh.r; ++j) {
    Index ix = j - 1 < pivot_x ? j - 1 : j;
    Index iv = j - 1 < pivot_v ? j - 1 : j;
    X(ix, j) = 1.0;
    V(iv, j) = 1.0;
  };
  Ensign::Matrix::blas_ops blas;
  Ensign::orthogonalize QR(&blas);
  // The quadrature weights define the discrete inner products.
  QR.gram_schmidt(X, Rx, mesh.h_xx[0]);
  QR.gram_schmidt(V, Rv, mesh.h_vv[0]);
  // Account for any normalization of the first columns during QR.
  S(0, 0) = Sx * Sv * Rx(0, 0) * Rv(0, 0);

  std::vector<const double*> X0, V0;
  X0.push_back(fx.begin());
  V0.push_back(fv.begin());

  stloc sl = (device == "CPU") ? stloc::host : stloc::device;
  Ensign::lr2<double> low_rank_solver(mesh.r, {mesh.dxx_mult, mesh.dvv_mult}, sl);
  std::function<double(double*,double*)> ip_xx = 
    Ensign::inner_product_from_const_weight(mesh.h_xx[0], mesh.dxx_mult);
  std::function<double(double*,double*)> ip_vv = 
    Ensign::inner_product_from_const_weight(mesh.h_vv[0], mesh.dvv_mult);
  initialize(low_rank_solver, X0, V0, ip_xx, ip_vv, blas);

  // C1, C2 compute
  coeff_C compute_C(sl, mesh);
  mat C1, C2_plus_wind, C2_neg_wind;
  C1.resize({mesh.r, mesh.r});
  C2_plus_wind.resize({mesh.r, mesh.r});
  C2_neg_wind.resize({mesh.r, mesh.r});
  compute_C(low_rank_solver.V, C1, C2_plus_wind, C2_neg_wind, blas);

  // E compute 
  vec E({mesh.dxx_mult});
  electric_field compute_E(sl, mesh);
  
  // D1(E), D2 preparation
  coeff_D compute_D(sl, mesh);
  mat D1, D2_plus_wind, D2_neg_wind;

  return 0;
};

