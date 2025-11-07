#ifndef FHE16_THREAD_POOL_H
#define FHE16_THREAD_POOL_H

#include <pthread.h>

#ifndef __cplusplus
#include<stdint.h>
#endif


#include<CMAKEPARAM.h>

#if CPUTYPE == 1 // IF LINUX
    #include<stdint.h>
    #include<stdbool.h>

#endif


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int *buffer;
	_Atomic int head;
	_Atomic int tail;
} LockfreeQUEUE;






typedef struct {
	void (*function)(void*);
	void* arg;
	bool TERMINATE;
	bool LAST_WORK;
} task_t;

typedef struct {
	task_t *tasks;
	int front, rear, count, queue_size;
	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
	pthread_cond_t		empty_cond;


} task_queue_t;

typedef struct {
	pthread_t *threads;
	task_queue_t *task_queue;
	volatile int *awaken;
	int thread_num;
	int queue_size;
	volatile int stop;
} thread_pool_t;

typedef struct{
	thread_pool_t *pool;
	volatile int *awaken;
	int id;
} tmp_data_t;


thread_pool_t *thread_pool_init(int Thread_num, int QUEUE_SIZE);
void thread_pool_destroy(thread_pool_t *pool);
void thread_queue_push(task_queue_t *q, void (*function)(void *), void* arg );
void task_queue_push(task_queue_t* q, void (*function)(void*), void* arg);
void waiting_all_sleep(thread_pool_t *pool);
void check_q_empty(thread_pool_t *pool);



//thread_pool_t *FHE16_POOL;

#ifdef __cplusplus
}
#endif



#endif
