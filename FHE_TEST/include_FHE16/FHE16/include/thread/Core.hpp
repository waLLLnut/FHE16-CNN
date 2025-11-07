#ifndef FHE16_CORECORE_H
#define FHE16_CORECORE_H

#include<atomic>
#include<pthread.h>
//#include<BINFHE.hpp>

#include<BinOperationCstyle.hpp>

typedef struct {
    int socket;
    int core;
    int cpu_id;
} PhysicalCore;


int get_physical_core_count();
int get_core_id();

//int get_physical_cpus(int* phys_cpus, int max);

int is_duplicate(PhysicalCore list[], int count, int socket, int core);
int get_physical_cores(PhysicalCore cores[], int max);

int16_t ***alloc_3d(int D, int n);
void init_upper_triangle_twos(int16_t **A0, int n);
int layer_3to2_ordered(int16_t **in, int16_t **out, int n, int in_rows);

int dadda_next_target_below(int h);
void dadda_active_rows(int n, int16_t *out, int *out_len);
int wallace_next_target_below(int h);
void wallace_active_rows(int n, int16_t *out, int *out_len);
void MakeA(int16_t ***ARR, int n, int16_t *actives, int alen);

void print_matrix(const char* tag, int16_t **A, int rows, int n);




void NAF_Calculation(int *in, int *out, int nbits);
void NAF_Calculation_signed(int *in, int *out, int nbits);
#endif



