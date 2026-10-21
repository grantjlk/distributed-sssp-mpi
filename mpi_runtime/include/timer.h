#pragma once
#include <mpi.h>

// Wall-clock timer for collective MPI regions.
// start() barriers so every rank begins together, and stop() returns the
// slowest rank's elapsed time (the real runtime of a parallel step).
class MpiTimer {
public:
    void start() {
        MPI_Barrier(MPI_COMM_WORLD);
        t0_ = MPI_Wtime();
    }

    // collective: every rank must call it, and every rank gets the max
    double stop() const {
        double local  = MPI_Wtime() - t0_;
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return global;
    }

private:
    double t0_ = 0.0;
};
