#pragma once
#include <mpi.h>
#include <type_traits>
#include <vector>

// Personalized all-to-all exchange: sends outbox[r] to rank r for every r and
// returns everything this rank received, concatenated in source-rank order.
// Collective: every rank in comm must call it in the same step.
//
// remote_bytes is increased by the bytes this rank sent to *other* ranks
// (the self-send is a local copy, not network traffic).
template <typename T>
std::vector<T> exchange(const std::vector<std::vector<T>>& outbox, MPI_Comm comm,
                        long long& remote_bytes)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "exchange() sends raw bytes, so T must be trivially copyable");

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. tell every rank how many elements to expect from us
    std::vector<int> send_counts(size), recv_counts(size);
    for (int r = 0; r < size; r++) {
        send_counts[r] = (int)outbox[r].size();
    }
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, comm);

    // 2. displacements: where each rank's chunk starts in the flat buffers
    std::vector<int> send_displs(size, 0), recv_displs(size, 0);
    for (int r = 1; r < size; r++) {
        send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }

    std::vector<T> send_buf;
    send_buf.reserve(send_displs[size - 1] + send_counts[size - 1]);
    for (int r = 0; r < size; r++) {
        send_buf.insert(send_buf.end(), outbox[r].begin(), outbox[r].end());
        if (r != rank) {
            remote_bytes += (long long)outbox[r].size() * sizeof(T);
        }
    }
    std::vector<T> recv_buf(recv_displs[size - 1] + recv_counts[size - 1]);

    // 3. one element of T = sizeof(T) bytes, so counts stay in elements
    MPI_Datatype elem;
    MPI_Type_contiguous((int)sizeof(T), MPI_BYTE, &elem);
    MPI_Type_commit(&elem);

    int rc = MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), elem,
                           recv_buf.data(), recv_counts.data(), recv_displs.data(), elem,
                           comm);
    MPI_Type_free(&elem);
    if (rc != MPI_SUCCESS) MPI_Abort(comm, rc);

    return recv_buf;
}
