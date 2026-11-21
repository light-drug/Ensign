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

template<size_t d>
struct grid_info {
  Index r;
  mind<d> N_xx; // x维度网格点数量
  mind<d> N_vv; // v维度网格点数量
  mfp<2*d> lim_xx, lim_vv;
  mfp<d>   h_xx, h_vv;
  Index dxx_mult, dvv_mult, dxxh_mult, dvvh_mult;

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
  
    dxxh_mult = N_xx[2]*N_xx[1]*(N_xx[0]/2 + 1);
    dvvh_mult = N_vv[2]*N_vv[1]*(N_vv[0]/2 + 1);
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

int main(int argc, char** argv)
{
  constexpr int dim = 1;
  constexpr int dim2 = 2 * dim;
  std::string device = "cpu";
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
  options.AddOption(&device, "-device", "--device", "Simulation device (cpu or gpu)");
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

  mind<dim> N_xx = {nx};
  mind<dim> N_vv = {nv};  
  mfp<dim2> lim_xx = {x1, x2}; // Limits for box [x1,x2] {x1,x2}
  mfp<dim2> lim_vv = {v1, v2}; // Limits for box [v1,v2] {v1,v2}
  Index r = rank;
  grid_info<dim> mesh(r, N_xx, N_vv, lim_xx, lim_vv);

  vec fx({mesh.dxx_mult});
  std::cout << "dxx_mult: " << mesh.dxx_mult << std::endl;
  // 此处传入的i为该点位置
  componentwise_vec_omp(mesh.N_xx, [&fx, &mesh](Index idx, mind<dim> i) {
    double delta1 = 0.01;
    double kappa1 = 0.5;
    mfp<dim> x  = mesh.x(i);
    fx(idx) = 1.0 + delta1 * std::cos(kappa1*x[0]);
  });

  vec fv({mesh.dvv_mult});
  std::cout << "dvv_mult: " << mesh.dvv_mult << std::endl;
  componentwise_vec_omp(mesh.N_vv, [&fv, &mesh](Index idx, mind<dim> i) {
    mfp<dim> v  = mesh.v(i);
    fv(idx) = (1.0/(sqrt(pow(2*M_PI, dim)))) * 
              exp(-(std::pow(v[0], 2)) / 2.0);
  });

  vector<const double*> X, V;
  X.push_back(fx.begin());
  V.push_back(fv.begin());

  Ensign::Matrix::blas_ops blas;
  Ensign::orthogonalize QR(&blas);
  
  return 0;
}

// struct coeff_C {

//   coeff_C(stloc sl, grid_info<1> _mesh) : gi(_gi), tmpVhat(sl) {

//     tmpVhat.resize({gi.dvvh_mult,gi.r});

//     we       = create_vec_array(gi.dvv_mult, sl);
//     dVhat    = create_cmat_array({gi.dvvh_mult,gi.r}, sl);
//     dV       = create_mat_array({gi.dvv_mult,gi.r}, sl);
//     h_lambda_n = create_cvec_array(gi.dvvh_mult, stloc::host);

//     // Initialize we
//     array<vec,3> h_we = create_vec_array(gi.dvv_mult, stloc::host);
//     array<vec,3> v    = create_vec_array({gi.dvv_mult}, stloc::host);

//     componentwise_vec_omp(gi.N_vv, [this, &v](Index idx, mind<3> i) {
//       v[0](idx) = gi.v(0, i[0]);
//       v[1](idx) = gi.v(1, i[1]);
//       v[2](idx) = gi.v(2, i[2]);
//     });

//     #ifdef __OPENMP__
//     #pragma omp parallel for
//     #endif
//     for(Index j = 0; j < gi.dvv_mult; j++){
//       h_we[0](j) = v[0](j) * gi.h_vv[0] * gi.h_vv[1] * gi.h_vv[2];
//       h_we[1](j) = v[1](j) * gi.h_vv[0] * gi.h_vv[1] * gi.h_vv[2];
//       h_we[2](j) = v[2](j) * gi.h_vv[0] * gi.h_vv[1] * gi.h_vv[2];
//     }

//     we[0] = h_we[0];
//     we[1] = h_we[1];
//     we[2] = h_we[2];

//     double ncvv = 1.0 / (gi.dvv_mult);
//     componentwise_vec_fourier_omp(gi.N_vv, [this, ncvv](Index idx, mind<3> i) {
//       Index mult_j = freq(i[1], gi.N_vv[1]);
//       Index mult_k = freq(i[2], gi.N_vv[2]);
//       h_lambda_n[0](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[1]-gi.lim_vv[0])*i[0])*ncvv;
//       h_lambda_n[1](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[3]-gi.lim_vv[2])*mult_j)*ncvv;
//       h_lambda_n[2](idx) = complex<double>(0.0,2.0*M_PI/(gi.lim_vv[5]-gi.lim_vv[4])*mult_k)*ncvv;
//     });

//       #ifdef __CUDA__
//       if(sl == stloc::device) {
//         d_lim_vv = make_unique_ptr<vec>(array<Index,1>({6}), stloc::device);
//         cudaMemcpy(d_lim_vv->data(), gi.lim_vv.data(), 6*sizeof(double), cudaMemcpyHostToDevice);
//       }
//       #endif
//   }

//   void operator()(mat& V, array<mat,3>& C1, array<mat,3>& C2, array<cmat,3>& C2c, const blas_ops& blas) {

//       // C1
//       coeff(V, V, we[0], C1[0], blas);
//       coeff(V, V, we[1], C1[1], blas);
//       coeff(V, V, we[2], C1[2], blas);

//       // C2
//       if(fft == nullptr)
//         fft = make_unique_ptr<fft3d<2>>(gi.N_xx, V, tmpVhat);

//       // TODO: relies on V not being overwritten
//       fft->forward(V, tmpVhat);

//       if(V.sl == stloc::host) {
//         ptw_mult_row(tmpVhat,h_lambda_n[0],dVhat[0]);
//         ptw_mult_row(tmpVhat,h_lambda_n[1],dVhat[1]);
//         ptw_mult_row(tmpVhat,h_lambda_n[2],dVhat[2]);
//       } else {
//         #ifdef __CUDA__
//         double ncvv = 1.0 / (gi.dvv_mult);
//         ptw_mult_row_cplx_fourier_3d<<<(gi.dvvh_mult*gi.r+n_threads-1)/n_threads,n_threads>>>(gi.dvvh_mult*gi.r, gi.N_vv[0]/2+1, gi.N_vv[1], gi.N_vv[2], (cuDoubleComplex*)tmpVhat.begin(), d_lim_vv->data(), ncvv, (cuDoubleComplex*)dVhat[0].begin(), (cuDoubleComplex*)dVhat[1].begin(), (cuDoubleComplex*)dVhat[2].begin());
//         #endif
//       }

//       fft->backward(dVhat[0], dV[0]);
//       fft->backward(dVhat[1], dV[1]);
//       fft->backward(dVhat[2], dV[2]);

//       coeff(V, dV[0], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[0], blas);
//       coeff(V, dV[1], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[1], blas);
//       coeff(V, dV[2], gi.h_vv[0]*gi.h_vv[1]*gi.h_vv[2], C2[2], blas);

//       if(V.sl == stloc::host) {
//         C2[0].to_cplx(C2c[0]);
//         C2[1].to_cplx(C2c[1]);
//         C2[2].to_cplx(C2c[2]);
//       } else {
//         #ifdef __CUDA__
//         cplx_conv<<<(C2[0].num_elements()+n_threads-1)/n_threads,n_threads>>>(C2[0].num_elements(), C2[0].begin(), (cuDoubleComplex*)C2c[0].begin());
//         cplx_conv<<<(C2[1].num_elements()+n_threads-1)/n_threads,n_threads>>>(C2[1].num_elements(), C2[1].begin(), (cuDoubleComplex*)C2c[1].begin());
//         cplx_conv<<<(C2[2].num_elements()+n_threads-1)/n_threads,n_threads>>>(C2[2].num_elements(), C2[2].begin(), (cuDoubleComplex*)C2c[2].begin());
//         #endif
//       }
//   }

// private:
//   grid_info<3> gi;
//   array<vec,3> we;
//   std::unique_ptr<fft3d<2>> fft;
//   cmat tmpVhat;
//   array<cmat,3> dVhat;
//   array<mat,3> dV;
//   array<cvec,3> h_lambda_n;
//   std::unique_ptr<vec> d_lim_vv;
// };

