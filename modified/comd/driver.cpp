#include "interface.h"
#include "utility.h" 

#define TEST_ROUND 9

// CUDA wrappers
extern "C" void advance_position(sim_t sim_D);
extern "C" void advance_velocity(sim_t sim_D);

extern "C" void lj_force(sim_t sim_D);

extern "C" void eam_force_1(const char *method, sim_t sim_D);
extern "C" void eam_force_2(sim_t sim_D);
extern "C" void eam_force_3(const char *method, sim_t sim_D);

extern "C" void copy_halos(sim_t sim_D);

void init(const config &c, simflat_t *simflat, sim_t &d_sim)
{
  sim_t h_sim;

  // copy sim data on host
  create_sim_host(c, simflat, &h_sim);

  // dump data
  //dump_sim_info(h_sim);

  // allocate sim data on device
  create_sim_dev(h_sim, &d_sim);
  
  // fill sim data on device
  fill_sim_dev(h_sim, d_sim);
}

void done(sim_t &d_sim)
{
  // free sim data memory
  destroy_sim_dev(d_sim);
}

void compute_force(const config &c, const sim_t &sim, bool first)
{
  if (c.eam_flag) {
    if (!first) 
      copy_halos(sim);
      
    eam_force_1(c.method.c_str(), sim);
    eam_force_2(sim);

    copy_halos(sim);
    
    eam_force_3(c.method.c_str(), sim);
  }
  else {
    lj_force(sim);
  }
}

void run(simflat_t *simflat, const config &c, sim_t &sim)
{
  // initial compute
#ifndef ECX_TARGET
  compute_force(c, sim, true);
#else
  if (c.iters == 0 && c.steps == 0 && c.eam_flag)    	// single kernel run for ECX to reduce simulation time
    eam_force_1(c.method.c_str(), sim);
  else
    compute_force(c, sim, true);
#endif
    float elapsed_time, min_time = -1;

    for(int r = 0; r < TEST_ROUND; r++){
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin);

        for (int it = 0; it < c.iters; it++)
            for (int step = 0; step < c.steps; step++) 
            {
            // advance particle positions dt/2       
            advance_position(sim);

            // compute force
            compute_force(c, sim, false);

            // advance velocity a full timestep
            advance_velocity(sim);

            // advance particle positions dt/2
            advance_position(sim);
        }

        cudaEventRecord(end);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, begin, end);

        if(min_time < 0 || elapsed_time < min_time){
            min_time = elapsed_time;
        }

#ifdef SEPERATE
        printf("Seperate version. ");
#endif
        printf("elapsed time: %f ms, min time: %f ms\n", elapsed_time, min_time);

        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }


  // compute & print energy term
  compute_energy(simflat, sim);
  printf("Energy = %.20f\n", sim.energy); 

//   std::fstream fs("output-gpu.csv", std::ios::out);
//   fs << (c.eam_flag ? "EAM" : "LJ") << "-" << c.x << "-" << c.y << "-" << c.z << "-" << sim.total_atoms << ", time, " << us_atom << ", us/atom" << std::endl;
//   fs.close();
}
 
int main(int argc, char **argv)
{
  config c;		// config 
  simflat_t *simflat;	// flat data from cmd line
  sim_t sim; 		// prepared data for simulation run

  simflat = parse(argc, argv, c);
  init(c, simflat, sim);
  run(simflat, c, sim);
  done(sim);

  delete simflat;
  return 0;
}
