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

int main(int argc, char** argv)
{
  std::string device = "cpu";
  std::string problem = "ts";
  double final_time = 60.0;
  double deltat = 0.01;
  int rank = 10;
  double nx, nv;
  int omp_threads = -1;
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

  return 0;
}
